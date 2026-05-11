/**
 * API Key Authentication System
 * Manages API keys for GPT-compatible authentication
 */

import { v4 as uuidv4 } from 'uuid';
import { ApiKeyEntry } from './types';

export class ApiKeyStore {
  private keys: Map<string, ApiKeyEntry>;
  private mutex: Promise<void>;

  constructor() {
    this.keys = new Map();
    this.mutex = Promise.resolve();
    
    // Add default key
    this.addKey('sk-ds4lite-default', 'default');
  }

  /**
   * Validate an API key and increment request counter
   */
  async validate(key: string): Promise<boolean> {
    if (!key || key.length === 0) return false;

    const entry = this.keys.get(key);
    if (entry && entry.active) {
      entry.requests++;
      return true;
    }
    return false;
  }

  /**
   * Add a new API key
   */
  async addKey(key: string, name: string = 'unnamed'): Promise<boolean> {
    if (this.keys.size >= 100) return false;

    const entry: ApiKeyEntry = {
      key,
      name,
      requests: 0,
      tokensUsed: 0,
      created: new Date(),
      active: true,
    };

    this.keys.set(key, entry);
    return true;
  }

  /**
   * Generate and add a new random API key
   */
  async generateKey(name: string = 'unnamed'): Promise<string> {
    const key = `sk-${uuidv4()}`;
    await this.addKey(key, name);
    return key;
  }

  /**
   * Revoke an API key
   */
  async revokeKey(key: string): Promise<boolean> {
    const entry = this.keys.get(key);
    if (entry) {
      entry.active = false;
      return true;
    }
    return false;
  }

  /**
   * List all API keys (without exposing full key for security)
   */
  async listKeys(): Promise<ApiKeyEntry[]> {
    return Array.from(this.keys.values()).map(entry => ({
      ...entry,
      key: entry.key === 'sk-ds4lite-default' ? entry.key : `${entry.key.substring(0, 8)}...${entry.key.substring(entry.key.length - 4)}`,
    }));
  }

  /**
   * Get key statistics
   */
  async getKeyStats(key: string): Promise<ApiKeyEntry | null> {
    const entry = this.keys.get(key);
    if (entry) {
      return { ...entry };
    }
    return null;
  }

  /**
   * Update token usage for a key
   */
  async updateTokenUsage(key: string, tokens: number): Promise<void> {
    const entry = this.keys.get(key);
    if (entry) {
      entry.tokensUsed += tokens;
    }
  }

  /**
   * Export keys to JSON (for persistence)
   */
  exportKeys(): string {
    const data = Array.from(this.keys.entries()).map(([key, entry]) => ({
      key,
      name: entry.name,
      requests: entry.requests,
      tokensUsed: entry.tokensUsed,
      created: entry.created.toISOString(),
      active: entry.active,
    }));
    return JSON.stringify(data, null, 2);
  }

  /**
   * Import keys from JSON (for persistence)
   */
  importKeys(json: string): boolean {
    try {
      const data = JSON.parse(json);
      if (!Array.isArray(data)) return false;

      for (const item of data) {
        if (!item.key || !item.name) continue;
        
        const entry: ApiKeyEntry = {
          key: item.key,
          name: item.name,
          requests: item.requests || 0,
          tokensUsed: item.tokensUsed || 0,
          created: new Date(item.created || Date.now()),
          active: item.active !== false,
        };
        this.keys.set(item.key, entry);
      }
      return true;
    } catch {
      return false;
    }
  }
}

// Global instance
export const globalApiKeyStore = new ApiKeyStore();
