/**
 * DS4 Lite Server - Main Application
 * Condensed web server for low-bandwidth inference with statistical range compression
 */

import express, { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import { createGzip } from 'zlib';
import { globalApiKeyStore } from './auth';
import { gzipCompress } from './compression';
import {
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatCompletionChunkResponse,
  ModelsListResponse,
  HealthResponse,
  AdminKeyAction,
  ServerConfig,
} from './types';

const VERSION = '1.0.0';

export class DS4LiteServer {
  private app: express.Application;
  private config: ServerConfig;
  private generationMutex: Promise<void>;

  constructor(config?: Partial<ServerConfig>) {
    this.app = express();
    this.generationMutex = Promise.resolve();
    
    // Default configuration optimized for low-bandwidth hardware
    this.config = {
      port: config?.port ?? 8080,
      host: config?.host ?? '0.0.0.0',
      ctxSize: config?.ctxSize ?? 32768,
      modelPath: config?.modelPath ?? './ds4flash.gguf',
      defaultApiKey: 'sk-ds4lite-default',
    };

    this.setupMiddleware();
    this.setupRoutes();
  }

  /**
   * Setup Express middleware
   */
  private setupMiddleware(): void {
    // CORS for cross-origin requests
    this.app.use(cors());
    
    // JSON body parser with size limit
    this.app.use(express.json({ limit: '10mb' }));
    
    // Text body parser for non-JSON requests
    this.app.use(express.text({ limit: '10mb' }));
  }

  /**
   * Setup API routes
   */
  private setupRoutes(): void {
    // Health check endpoint (no auth required)
    this.app.get('/health', this.handleHealth.bind(this));
    
    // OpenAI-compatible endpoints
    this.app.get('/v1/models', this.authenticate.bind(this), this.handleModels.bind(this));
    this.app.post('/v1/chat/completions', this.authenticate.bind(this), this.handleChatCompletions.bind(this));
    
    // Admin endpoints for key management
    this.app.post('/admin/keys', this.authenticate.bind(this), this.handleAdminKeys.bind(this));
    
    // 404 handler
    this.app.use((req: Request, res: Response) => {
      res.status(404).json({
        error: {
          message: 'Not Found',
          type: 'invalid_request_error',
        },
      });
    });
  }

  /**
   * Authentication middleware
   */
  private async authenticate(req: Request, res: Response, next: NextFunction): Promise<void> {
    const authHeader = req.headers.authorization;
    
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      res.status(401).json({
        error: {
          message: 'Unauthorized - Missing or invalid Authorization header',
          type: 'authentication_error',
        },
      });
      return;
    }

    const apiKey = authHeader.substring(7); // Remove 'Bearer ' prefix
    
    if (!(await globalApiKeyStore.validate(apiKey))) {
      res.status(401).json({
        error: {
          message: 'Unauthorized - Invalid API key',
          type: 'authentication_error',
        },
      });
      return;
    }

    next();
  }

  /**
   * Health check endpoint
   */
  private handleHealth(req: Request, res: Response): void {
    const response: HealthResponse = {
      status: 'healthy',
      version: VERSION,
    };
    res.json(response);
  }

  /**
   * Models list endpoint (OpenAI-compatible)
   */
  private handleModels(req: Request, res: Response): void {
    const response: ModelsListResponse = {
      object: 'list',
      data: [
        {
          id: 'deepseek-v4-flash',
          object: 'model',
          created: Math.floor(Date.now() / 1000),
          owned_by: 'ds4-lite',
        },
      ],
    };
    res.json(response);
  }

  /**
   * Chat completions endpoint (OpenAI-compatible)
   */
  private async handleChatCompletions(req: Request, res: Response): Promise<void> {
    const body = req.body as ChatCompletionRequest;
    
    // Extract parameters with defaults
    const messages = body.messages ?? [];
    const maxTokens = body.max_tokens ?? 256;
    const temperature = body.temperature ?? 0.7;
    const stream = body.stream ?? false;

    // Extract user message from messages array
    let prompt = 'Hello!';
    for (const msg of messages) {
      if (msg.role === 'user' && msg.content) {
        prompt = msg.content;
        break;
      }
    }

    // Simulate token counts
    const promptTokens = Math.ceil(prompt.length / 4);
    const completionTokens = Math.min(maxTokens, Math.ceil(promptTokens * 0.8));

    if (stream) {
      // Streaming response
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      const chunkId = `chatcmpl-${Date.now()}`;
      const timestamp = Math.floor(Date.now() / 1000);

      // Send initial chunk with role
      const initialChunk: ChatCompletionChunkResponse = {
        id: chunkId,
        object: 'chat.completion.chunk',
        created: timestamp,
        model: 'deepseek-v4-flash',
        choices: [
          {
            index: 0,
            delta: { role: 'assistant', content: '' },
            finish_reason: null,
          },
        ],
      };
      res.write(`data: ${JSON.stringify(initialChunk)}\n\n`);

      // Simulate streaming content in chunks
      const simulatedResponse = `This is a simulated response to: "${prompt.substring(0, 50)}${prompt.length > 50 ? '...' : ''}". In a real implementation, this would connect to the DS4 inference engine.`;
      
      const words = simulatedResponse.split(' ');
      for (let i = 0; i < words.length; i += 2) {
        const chunk: ChatCompletionChunkResponse = {
          id: chunkId,
          object: 'chat.completion.chunk',
          created: timestamp,
          model: 'deepseek-v4-flash',
          choices: [
            {
              index: 0,
              delta: { content: words.slice(i, i + 2).join(' ') + ' ' },
              finish_reason: null,
            },
          ],
        };
        res.write(`data: ${JSON.stringify(chunk)}\n\n`);
        await new Promise(resolve => setTimeout(resolve, 50)); // Simulate delay
      }

      // Send final chunk
      const finalChunk: ChatCompletionChunkResponse = {
        id: chunkId,
        object: 'chat.completion.chunk',
        created: timestamp,
        model: 'deepseek-v4-flash',
        choices: [
          {
            index: 0,
            delta: {},
            finish_reason: 'stop',
          },
        ],
      };
      res.write(`data: ${JSON.stringify(finalChunk)}\n\n`);
      res.write('data: [DONE]\n\n');
      res.end();
    } else {
      // Non-streaming response
      const response: ChatCompletionResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: 'deepseek-v4-flash',
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: `This is a simulated response to: "${prompt.substring(0, 50)}${prompt.length > 50 ? '...' : ''}". In a real implementation, this would connect to the DS4 inference engine.`,
            },
            finish_reason: 'stop',
          },
        ],
        usage: {
          prompt_tokens: promptTokens,
          completion_tokens: completionTokens,
          total_tokens: promptTokens + completionTokens,
        },
      };

      // Apply GZIP compression for large responses (>1KB)
      const responseBody = JSON.stringify(response);
      const acceptEncoding = req.headers['accept-encoding'] || '';
      
      if (acceptEncoding.includes('gzip') && responseBody.length > 1024) {
        res.setHeader('Content-Encoding', 'gzip');
        const compressed = await gzipCompress(responseBody);
        res.send(compressed);
      } else {
        res.json(response);
      }
    }

    // Update token usage for the API key
    const authHeader = req.headers.authorization;
    if (authHeader && authHeader.startsWith('Bearer ')) {
      const apiKey = authHeader.substring(7);
      await globalApiKeyStore.updateTokenUsage(apiKey, promptTokens + completionTokens);
    }
  }

  /**
   * Admin endpoint for API key management
   */
  private async handleAdminKeys(req: Request, res: Response): Promise<void> {
    const body = req.body as AdminKeyAction;

    switch (body.action) {
      case 'add':
        if (body.key && body.name) {
          const success = await globalApiKeyStore.addKey(body.key, body.name);
          if (success) {
            res.json({ success: true, message: 'Key added successfully' });
          } else {
            res.status(400).json({ success: false, message: 'Failed to add key - maximum keys reached' });
          }
        } else {
          res.status(400).json({ success: false, message: 'Missing key or name parameter' });
        }
        break;

      case 'generate':
        const name = body.name ?? 'unnamed';
        const newKey = await globalApiKeyStore.generateKey(name);
        res.json({ success: true, key: newKey, name });
        break;

      case 'revoke':
        if (body.key) {
          const success = await globalApiKeyStore.revokeKey(body.key);
          if (success) {
            res.json({ success: true, message: 'Key revoked successfully' });
          } else {
            res.status(404).json({ success: false, message: 'Key not found' });
          }
        } else {
          res.status(400).json({ success: false, message: 'Missing key parameter' });
        }
        break;

      case 'list':
      default:
        const keys = await globalApiKeyStore.listKeys();
        res.json({ keys });
        break;
    }
  }

  /**
   * Start the server
   */
  public start(): void {
    const { host, port } = this.config;
    
    this.app.listen(port, host, () => {
      console.log(`╔════════════════════════════════════════════════════╗`);
      console.log(`║  DS4 Lite Server v${VERSION}                          ║`);
      console.log(`╠════════════════════════════════════════════════════╣`);
      console.log(`║  Listening on http://${host}:${port.toString().padEnd(5)}                     ║`);
      console.log(`║  Default API key: sk-ds4lite-default                ║`);
      console.log(`║  Context size: ${this.config.ctxSize.toString().padEnd(6)} tokens                   ║`);
      console.log(`╠════════════════════════════════════════════════════╣`);
      console.log(`║  Endpoints:                                        ║`);
      console.log(`║    GET  /health                                    ║`);
      console.log(`║    GET  /v1/models                                 ║`);
      console.log(`║    POST /v1/chat/completions                       ║`);
      console.log(`║    POST /admin/keys                                ║`);
      console.log(`╚════════════════════════════════════════════════════╝`);
      console.log('');
      console.log('Example usage:');
      console.log('  curl http://localhost:' + port + '/health');
      console.log('  curl http://localhost:' + port + '/v1/models -H "Authorization: Bearer sk-ds4lite-default"');
      console.log('  curl http://localhost:' + port + '/v1/chat/completions \\');
      console.log('    -H "Authorization: Bearer sk-ds4lite-default" \\');
      console.log('    -H "Content-Type: application/json" \\');
      console.log('    -d \'{"messages":[{"role":"user","content":"Hello"}]}\'');
    });
  }

  /**
   * Stop the server
   */
  public stop(): Promise<void> {
    return new Promise((resolve) => {
      const server = this.app.listen(undefined, () => {
        server.close(() => {
          console.log('Server stopped');
          resolve();
        });
      });
    });
  }
}

// Export singleton instance for CLI usage
export let serverInstance: DS4LiteServer | null = null;

/**
 * Main entry point when run directly
 */
if (require.main === module) {
  const args = process.argv.slice(2);
  const config: Partial<ServerConfig> = {};

  // Parse command line arguments
  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '-m':
      case '--model':
        if (args[i + 1]) config.modelPath = args[++i]!;
        break;
      case '-p':
      case '--port':
        if (args[i + 1]) config.port = parseInt(args[++i]!, 10);
        break;
      case '--ctx':
      case '--context':
        if (args[i + 1]) config.ctxSize = parseInt(args[++i]!, 10);
        break;
      case '-h':
      case '--host':
        if (args[i + 1]) config.host = args[++i]!;
        break;
      case '--help':
        console.log('DS4 Lite Server v' + VERSION);
        console.log('Usage: ts-node src/server.ts [options]');
        console.log('Options:');
        console.log('  -m, --model PATH     Model path (default: ./ds4flash.gguf)');
        console.log('  -p, --port PORT      Port to listen on (default: 8080)');
        console.log('  --ctx, --context N   Context size (default: 32768)');
        console.log('  -h, --host HOST      Host to bind (default: 0.0.0.0)');
        console.log('  --help               Show this help');
        process.exit(0);
    }
  }

  serverInstance = new DS4LiteServer(config);
  serverInstance.start();
}
