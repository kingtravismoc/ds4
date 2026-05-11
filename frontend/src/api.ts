import { Message, ChatCompletionResponse, ModelInfo, ApiKeyInfo } from './types';

const API_BASE = '';

export class ApiClient {
  private apiKey: string | null = null;

  setApiKey(key: string) {
    this.apiKey = key;
    localStorage.setItem('ds4_api_key', key);
  }

  getApiKey(): string | null {
    if (!this.apiKey) {
      this.apiKey = localStorage.getItem('ds4_api_key');
    }
    return this.apiKey;
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const apiKey = this.getApiKey();
    
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      'Accept-Encoding': 'gzip',
      ...options.headers,
    };

    if (apiKey) {
      headers['Authorization'] = `Bearer ${apiKey}`;
    }

    const response = await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      headers,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Request failed' }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return response.json();
  }

  async healthCheck(): Promise<{ status: string; model_loaded: boolean; timestamp: number }> {
    return this.request('/health');
  }

  async listModels(): Promise<ModelInfo[]> {
    const result = await this.request<{ object: string; data: ModelInfo[] }>('/v1/models');
    return result.data;
  }

  async chatCompletion(
    messages: Message[],
    options: {
      temperature?: number;
      max_tokens?: number;
      model?: string;
    } = {}
  ): Promise<ChatCompletionResponse> {
    return this.request('/v1/chat/completions', {
      method: 'POST',
      body: JSON.stringify({
        model: options.model || 'deepseek-v4-flash',
        messages,
        temperature: options.temperature ?? 0.7,
        max_tokens: options.max_tokens,
      }),
    });
  }

  async *chatCompletionStream(
    messages: Message[],
    options: {
      temperature?: number;
      max_tokens?: number;
      model?: string;
    } = {}
  ): AsyncGenerator<string> {
    const apiKey = this.getApiKey();
    
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      'Authorization': apiKey ? `Bearer ${apiKey}` : '',
    };

    const response = await fetch(`${API_BASE}/v1/chat/completions-stream`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        model: options.model || 'deepseek-v4-flash',
        messages,
        temperature: options.temperature ?? 0.7,
        max_tokens: options.max_tokens,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No response body');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') continue;
          
          try {
            const parsed = JSON.parse(data);
            if (parsed.choices?.[0]?.delta?.content) {
              yield parsed.choices[0].delta.content;
            }
          } catch {
            // Ignore parse errors
          }
        }
      }
    }
  }

  // Admin endpoints
  async createApiKey(name: string, role: string = 'user'): Promise<{ key: string }> {
    return this.request('/admin/keys/create', {
      method: 'POST',
      body: JSON.stringify({ name, role }),
    });
  }

  async deleteApiKey(key: string): Promise<{ success: boolean }> {
    return this.request('/admin/keys/delete', {
      method: 'POST',
      body: JSON.stringify({ key }),
    });
  }

  async listApiKeys(): Promise<ApiKeyInfo[]> {
    const result = await this.request<{ keys: ApiKeyInfo[] }>('/admin/keys/list');
    return result.keys;
  }
}

export const apiClient = new ApiClient();
