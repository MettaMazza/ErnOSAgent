# Provider Interface

The `Provider` trait (`src/provider/mod.rs`) is the universal abstraction for inference backends. Ern-OS ships with 3 implementations.

## Provider Trait

```rust
#[async_trait]
pub trait Provider: Send + Sync {
    fn id(&self) -> &str;
    fn display_name(&self) -> &str;
    async fn get_model_spec(&self) -> Result<ModelSpec>;
    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&serde_json::Value>,
        thinking: bool,
    ) -> Result<Receiver<StreamEvent>>;
    async fn chat_sync(
        &self,
        messages: &[Message],
        tools: Option<&serde_json::Value>,
    ) -> Result<String>;
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    async fn health(&self) -> bool;
}
```

### Method Purposes

| Method | Used By | Description |
|--------|---------|-------------|
| `chat()` | WebSocket handler | Streaming chat — returns `Receiver<StreamEvent>` |
| `chat_sync()` | Observer audit | Non-streaming — returns full text. Thinking disabled for latency |
| `get_model_spec()` | Startup | Auto-detects model name, context length, vision support |
| `embed()` | Memory embeddings | Generates embedding vector for text |
| `health()` | Startup health check | Returns true if backend is reachable |

## StreamEvent

```rust
pub enum StreamEvent {
    TextDelta(String),
    ThinkingDelta(String),
    ToolCalls(Vec<ToolCall>),
    Done,
    Error(String),
}
```

## Message

```rust
pub struct Message {
    pub role: String,
    pub content: Vec<ContentPart>,
}
```

Where `ContentPart` is:
```rust
pub enum ContentPart {
    Text(String),
    ImageUrl(String),
}
```

Factory methods:
- `Message::text(role, content)` — text-only message
- `Message::multipart(role, text, image_urls)` — text + images

## Implementations

### 1. LlamaCpp (`src/provider/llamacpp.rs`)

- **ID**: `"llamacpp"`
- **API**: OpenAI-compatible `/v1/chat/completions` (streaming via SSE)
- **Config**: `LlamaCppConfig` — server_binary, port, model_path, mmproj_path, n_gpu_layers
- **Features**: `build_server_args()` constructs the CLI arguments for `llama-server`
- **Health**: GET `/health` endpoint
- **Model spec**: GET `/v1/models` → parses model name and context length
- **Embedding**: Separate server on `embedding_port` via `llamacpp_embed.rs`

### 2. Ollama (`src/provider/ollama.rs`)

- **ID**: `"ollama"`
- **API**: Ollama native API (`/api/chat`, `/api/embed`)
- **Config**: `OllamaConfig` — base_url, model
- **Health**: GET `/api/tags` endpoint

### 3. OpenAI-Compatible (`src/provider/openai_compat.rs`)

- **ID**: `"openai_compat"`
- **API**: Standard OpenAI `/v1/chat/completions`
- **Config**: `OpenAICompatConfig` — base_url, api_key, model
- **Health**: GET `/v1/models` endpoint

## Stream Parser (`src/provider/stream_parser.rs`)

Parses Server-Sent Events (SSE) from the streaming HTTP response:

1. Reads `data: ` lines from the SSE stream
2. Parses each as JSON
3. Extracts delta content, tool calls, thinking tokens
4. Sends `StreamEvent` variants through the channel

## Provider Selection

`create_provider(config)` in `src/provider/mod.rs`:

```rust
match config.general.active_provider.as_str() {
    "llamacpp" => LlamaCppProvider::new(&config.llamacpp),
    "ollama" => OllamaProvider::new(&config.ollama),
    "openai_compat" => OpenAICompatProvider::new(&config.openai_compat),
    _ => error
}
```

Provider is selected at startup via `config.general.active_provider` and stored as `Arc<dyn Provider>` in `AppState`.

## ModelSpec

Auto-derived from the running provider at startup:

```rust
pub struct ModelSpec {
    pub name: String,
    pub context_length: usize,
    pub supports_vision: bool,
    pub supports_tool_calling: bool,
    pub supports_thinking: bool,
    pub embedding_dimensions: usize,
}
```

No model parameters are hardcoded. Context length, capabilities, and model name all come from the provider API.
