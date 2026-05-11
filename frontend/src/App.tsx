import { useState, useEffect, useRef } from 'react';
import { apiClient } from './api';
import { Message } from './types';
import './App.css';

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [apiKey, setApiKeyState] = useState(apiClient.getApiKey() || '');
  const [showSettings, setShowSettings] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(2048);
  const [healthStatus, setHealthStatus] = useState<{ status: string; model_loaded: boolean } | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const checkHealth = async () => {
    try {
      const health = await apiClient.healthCheck();
      setHealthStatus({ status: health.status, model_loaded: health.model_loaded });
    } catch (error) {
      setHealthStatus(null);
    }
  };

  const handleApiKeySave = () => {
    if (apiKey.trim()) {
      apiClient.setApiKey(apiKey.trim());
      setShowSettings(false);
      checkHealth();
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { role: 'user', content: input.trim() };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInput('');
    setIsLoading(true);

    try {
      const assistantMessage: Message = { role: 'assistant', content: '' };
      setMessages([...newMessages, assistantMessage]);

      // Use streaming
      const stream = apiClient.chatCompletionStream(newMessages, {
        temperature,
        max_tokens: maxTokens,
      });

      let accumulatedContent = '';
      for await (const token of stream) {
        accumulatedContent += token;
        setMessages([
          ...newMessages,
          { role: 'assistant', content: accumulatedContent },
        ]);
      }
    } catch (error) {
      const errorMessage: Message = {
        role: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
      };
      setMessages([...newMessages, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  return (
    <div className="app">
      <header className="header">
        <h1>DS4-Lite Inference</h1>
        <div className="header-controls">
          {healthStatus && (
            <span className={`status-badge ${healthStatus.model_loaded ? 'online' : 'offline'}`}>
              {healthStatus.model_loaded ? '● Model Ready' : '○ Model Loading'}
            </span>
          )}
          <button onClick={() => setShowSettings(!showSettings)} className="settings-btn">
            ⚙️ Settings
          </button>
        </div>
      </header>

      {showSettings && (
        <div className="settings-panel">
          <h3>Settings</h3>
          <div className="setting-group">
            <label>API Key</label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKeyState(e.target.value)}
              placeholder="sk-..."
            />
            <button onClick={handleApiKeySave}>Save</button>
          </div>
          <div className="setting-group">
            <label>Temperature: {temperature}</label>
            <input
              type="range"
              min="0"
              max="2"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
            />
          </div>
          <div className="setting-group">
            <label>Max Tokens: {maxTokens}</label>
            <input
              type="range"
              min="256"
              max="4096"
              step="256"
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value))}
            />
          </div>
        </div>
      )}

      <main className="chat-container">
        <div className="messages">
          {messages.length === 0 ? (
            <div className="welcome-message">
              <h2>Welcome to DS4-Lite</h2>
              <p>Start a conversation with the AI model</p>
              <p className="hint">Configure your API key in settings if needed</p>
            </div>
          ) : (
            messages.map((msg, index) => (
              <div key={index} className={`message ${msg.role}`}>
                <div className="message-avatar">
                  {msg.role === 'user' ? '👤' : '🤖'}
                </div>
                <div className="message-content">{msg.content}</div>
              </div>
            ))
          )}
          {isLoading && messages[messages.length - 1]?.role === 'user' && (
            <div className="message assistant">
              <div className="message-avatar">🤖</div>
              <div className="message-content">
                <span className="typing-indicator">Thinking...</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </main>

      <footer className="chat-input">
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading || !input.trim()}>
            Send
          </button>
          <button type="button" onClick={clearChat} disabled={messages.length === 0}>
            Clear
          </button>
        </form>
      </footer>
    </div>
  );
}

export default App;
