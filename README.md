# URL Research Platform v2.0 - Multi-Provider Edition

A comprehensive Python-based platform for analyzing URLs with **multi-provider LLM support**, automatic fallback, and advanced configuration management.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
copy .env.example .env

# Edit .env with your API keys
# Validate configuration

python config_validator.py validate
```


### 2. Start the Server

```bash
# Development server
uvicorn main_agnostic:app --reload

# Production server
2-5-10 Warrantr
uvicorn main_agnostic:app --host 0.0.0.0 --port 8000
```

### 3. Access the Platform

- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Provider Status**: http://localhost:8000/providers/status

## 🤖 Supported Providers

- **OpenAI** - GPT-3.5, GPT-4 models
- **Anthropic** - Claude-3 models
- **Ollama** - Local LLMs (Llama2, Mistral, etc.)
- **Google Gemini** - Gemini Pro models
- **Cohere** - Command series models
- **Azure OpenAI** - Enterprise deployment
- **Hugging Face** - Local transformers

## 🔧 Configuration

### Environment Variables

```bash
# OpenAI
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_PRIORITY=10

# Anthropic
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-3-haiku-20240307
ANTHROPIC_PRIORITY=20

# Ollama (Local)
OLLAMA_ENABLED=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
OLLAMA_PRIORITY=30
```

### Priority System

Lower numbers = higher priority. The system tries providers in priority order with automatic fallback.

## 🌐 Usage Examples

### Web Interface

1. Enter URLs (JSON or line-separated)
2. Select provider (or use auto-fallback)
3. Configure summary settings
4. View results with provider information

### API Usage

```python
import httpx

# Process URLs with specific provider
response = await httpx.post("http://localhost:8000/process-urls", json={
    "urls": [{"url": "https://example.com"}],
    "provider": "anthropic",
    "max_length": 150
})
```

### Docker Deployment

```bash
# Start all services
docker-compose up -d
# View logs
docker-compose logs -f
```

## 🔍 Features

✅ **Multi-Provider Support** - 7+ LLM providers
✅ **Automatic Fallback** - Never fails due to single provider issues
✅ **Cost Optimization** - Mix free local and paid cloud models
✅ **Real-time Monitoring** - Provider status and performance tracking
✅ **Custom Prompts** - Template system with placeholders
✅ **Batch Processing** - Concurrent URL processing
✅ **Configuration Management** - Validation and setup tools
✅ **Docker Support** - Ready for containerized deployment

## 📊 Provider Comparison

| Provider | Speed | Quality | Cost | Local |
|----------|-------|---------|------|-------|
| OpenAI GPT-3.5 | ⚡⚡⚡ | ⚡⚡⚡⚡ | ⚡⚡ | ❌ |
| Anthropic Claude | ⚡⚡ | ⚡⚡⚡⚡⚡ | ⚡⚡⚡ | ❌ |
| Ollama | ⚡ | ⚡⚡⚡ | ⚡⚡⚡⚡⚡ | ✅ |

## 🛠️ Development

### Project Structure

```markdown
url-research-platform-v2/
├── main_agnostic.py          # Core application
├── summary_providers.py      # Provider implementations
├── config_validator.py       # Configuration management
├── requirements_updated.txt  # Dependencies
├── .env.example             # Environment template
├── docker-compose.yml       # Container orchestration
└── docs/                    # Documentation
```

### Adding New Providers

1. Implement BaseSummaryProvider interface
2. Add to SummaryProviderFactory
3. Update configuration validation
4. Add environment variable support

## 📈 Monitoring

### Provider Status

```bash
# Check provider availability
curl http://localhost:8000/providers/status
```

### Performance Metrics

- Response times per provider
- Token usage tracking
- Error rates and fallback statistics
- Cost optimization insights

## 🔧 Troubleshooting

### Common Issues

- **Provider not available**: Check API keys and configuration
- **Slow performance**: Adjust concurrency settings or use faster providers
- **High costs**: Enable local providers (Ollama) for bulk processing

### Configuration Validation

```bash
python config_validator.py validate
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.

---
**Ready to research URLs with multi-provider power?** 🚀
