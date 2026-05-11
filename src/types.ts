/**
 * Type definitions for DS4 Lite Server
 */

export interface ApiKeyEntry {
  key: string;
  name: string;
  requests: number;
  tokensUsed: number;
  created: Date;
  active: boolean;
}

export interface RangeCompressed {
  minVal: number;
  maxVal: number;
  compressedData: Uint16Array;
  dataLen: number;
  origLen: number;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface ChatCompletionRequest {
  messages: ChatMessage[];
  model?: string;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stream?: boolean;
  stop?: string | string[];
}

export interface ChatCompletionChoice {
  index: number;
  message: {
    role: string;
    content: string;
  };
  finish_reason: string;
}

export interface ChatCompletionChunkChoice {
  index: number;
  delta: {
    role?: string;
    content?: string;
  };
  finish_reason?: string | null;
}

export interface UsageStats {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface ChatCompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: ChatCompletionChoice[];
  usage: UsageStats;
}

export interface ChatCompletionChunkResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: ChatCompletionChunkChoice[];
}

export interface ModelInfo {
  id: string;
  object: string;
  created: number;
  owned_by: string;
}

export interface ModelsListResponse {
  object: string;
  data: ModelInfo[];
}

export interface HealthResponse {
  status: string;
  version: string;
}

export interface AdminKeyAction {
  action: 'add' | 'list' | 'revoke' | 'generate';
  key?: string;
  name?: string;
}

export interface ServerConfig {
  port: number;
  host: string;
  ctxSize: number;
  modelPath: string;
  defaultApiKey: string;
}
