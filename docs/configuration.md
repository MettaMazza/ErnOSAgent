# Configuration Reference

Ern-OS reads `ern-os.toml` from the project root. If absent, all defaults apply.

## `[general]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `active_provider` | String | `"llamacpp"` | Provider backend: `"llamacpp"`, `"ollama"`, or `"openai_compat"` |
| `data_dir` | Path | `"data"` | Directory for all persistent data (memory, sessions, logs) |
| `kokoro_port` | Option\<u16\> | `8880` | Port for local Kokoro TTS server (auto-started) |
| `flux_port` | Option\<u16\> | `8890` | Port for local Flux image generation server (auto-started) |
| `whisper_port` | Option\<u16\> | `8891` | Port for local Whisper STT server (voice calls) |

## `[llamacpp]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `server_binary` | String | `"llama-server"` | Path to the llama-server binary |
| `port` | u16 | `8080` | Port for the inference server |
| `model_path` | String | `"./models/gemma-4-31B-it-Q4_K_M.gguf"` | Path to the GGUF model file |
| `mmproj_path` | Option\<String\> | `"./models/mmproj-F16.gguf"` | Multimodal projector for vision (optional) |
| `n_gpu_layers` | i32 | `999` | GPU layers to offload (-1 or 999 = all) |
| `embedding_port` | u16 | `8081` | Port for the embedding server |
| `embedding_model` | Option\<String\> | `None` | Separate embedding model path (uses main model if None) |
| `visual_token_budget` | usize | `560` | Visual token budget (70, 140, 280, 560, 1120) |
| `lora_adapter` | Option\<String\> | `None` | LoRA adapter GGUF to load at inference (incremental learning) |

## `[ollama]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `base_url` | String | `"http://localhost:11434"` | Ollama API base URL |
| `model` | String | `"gemma4:26b"` | Model tag to use |

## `[openai_compat]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `base_url` | String | `"http://localhost:1234/v1"` | OpenAI-compatible API base URL |
| `api_key` | Option\<String\> | `None` | API key (optional for local servers) |
| `model` | String | `"gemma-4-26b-it"` | Model identifier |

## `[observer]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable the observer audit system |

## `[web]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `port` | u16 | `3000` | WebUI server port |
| `open_browser` | bool | `true` | Auto-open browser on startup |

## `[prompt]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `system_prompt` | String | `"You are a helpful..."` | Base system prompt |
| `thinking_enabled` | bool | `true` | Enable thinking/reasoning mode |

## `[codes]`

Controls the integrated VS Code IDE (code-server).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable the code-server IDE |
| `port` | u16 | `8443` | Port for code-server |
| `workspace` | String | `"."` | Default workspace directory to open |

code-server is auto-detected at `~/.ernos/code-server-4.116.0-macos-arm64/bin/code-server`. If the binary is not found, the Codes tab shows a download prompt. The extension `ernos-ai` is pre-installed at `~/.ernos/code-server-4.116.0-macos-arm64/extensions/ernos-ai/`.

## `[tts]`

Controls the Kokoro text-to-speech engine.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable TTS |
| `port` | u16 | `8880` | Port for the Kokoro TTS server |

TTS is auto-started via `~/.ernos/sandbox/scripts/start-kokoro.py` using the Python environment at `~/.ernos/python/bin/python3.12`. The ONNX model runs locally with no external API calls.

## Sidecar Services (Auto-Started)

Ern-OS auto-starts three sidecar services on boot:

| Service | Default Port | Script | Purpose |
|---------|-------------|--------|---------|
| **Kokoro TTS** | 8880 | `scripts/start-kokoro.py` | Text-to-speech for message playback and voice calls |
| **Flux Image** | 8890 | `scripts/flux_server.py` | Local image generation via Flux model |
| **Whisper STT** | 8891 | (planned) | Speech-to-text for voice call input |

All services are health-checked on startup — if already running, they are reused. If the startup script is not found, the service is silently disabled.

## Environment Variables

The config is loaded from `ern-os.toml`. API keys for web search providers can be set via the Settings UI (stored in `data/api_keys.json` and loaded into process env on startup):

| Variable | Used By |
|----------|---------|
| `BRAVE_API_KEY` | Brave Search (1st in waterfall) |
| `SERPER_API_KEY` | Serper (2nd) |
| `TAVILY_API_KEY` | Tavily (3rd) |
| `SERPAPI_API_KEY` | SerpAPI (4th) |

DuckDuckGo and Wikipedia require no API keys (free fallbacks).

## Example `ern-os.toml`

```toml
[general]
active_provider = "llamacpp"
data_dir = "data"
flux_port = 8890

[llamacpp]
server_binary = "/opt/homebrew/bin/llama-server"
port = 8080
model_path = "./models/gemma-4-31B-it-Q4_K_M.gguf"
mmproj_path = "./models/mmproj-F16.gguf"
n_gpu_layers = 999
embedding_port = 8081
visual_token_budget = 560

[observer]
enabled = true

[web]
port = 3000
open_browser = true

[codes]
enabled = true
port = 8443
workspace = "."

[prompt]
thinking_enabled = true
```
