"""
Abstract summary provider system supporting multiple LLM providers
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import httpx
import json
import asyncio
from dataclasses import dataclass
import os
from enum import Enum

class ProviderType(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    GEMINI = "gemini"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    AZURE_OPENAI = "azure_openai"
    LOCAL_TRANSFORMER = "local_transformer"

@dataclass
class SummaryRequest:
    """Request object for summary generation"""
    text_content: str
    title: str = ""
    max_length: int = 150
    temperature: float = 0.3
    custom_prompt: Optional[str] = None

@dataclass
class SummaryResponse:
    """Response object from summary generation"""
    summary: str
    provider: str
    model: str
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    processing_time: float = 0
    error: Optional[str] = None

class BaseSummaryProvider(ABC):
    """Abstract base class for summary providers"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_name = self.__class__.__name__

    @abstractmethod
    async def generate_summary(self, request: SummaryRequest) -> SummaryResponse:
        """Generate a summary from the given text content"""
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the provider is available and properly configured"""
        pass

    @abstractmethod
    def get_required_config(self) -> List[str]:
        """Return list of required configuration keys"""
        pass

    def _build_prompt(self, request: SummaryRequest) -> str:
        """Build the prompt for summary generation"""
        if request.custom_prompt:
            return request.custom_prompt.format(
                title=request.title,
                content=request.text_content[:3000]
            )

        return f"""
Please provide a concise summary (2-3 sentences) of the following web page content:

Title: {request.title}

Content: {request.text_content[:3000]}

Focus on the main purpose, key information, and target audience of the page.
""".strip()

class OpenAIProvider(BaseSummaryProvider):
    """OpenAI GPT provider"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.model = config.get("model", "gpt-3.5-turbo")
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.client = None

    async def _get_client(self):
        if not self.client:
            import openai
            self.client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self.client

    async def generate_summary(self, request: SummaryRequest) -> SummaryResponse:
        import time
        start_time = time.time()

        try:
            client = await self._get_client()
            prompt = self._build_prompt(request)

            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise, accurate summaries of web page content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=request.max_length,
                temperature=request.temperature
            )

            summary = response.choices[0].message.content.strip()

            return SummaryResponse(
                summary=summary,
                provider="OpenAI",
                model=self.model,
                tokens_used=response.usage.total_tokens if response.usage else None,
                processing_time=time.time() - start_time
            )

        except Exception as e:
            return SummaryResponse(
                summary="",
                provider="OpenAI",
                model=self.model,
                error=str(e),
                processing_time=time.time() - start_time
            )

    async def is_available(self) -> bool:
        return bool(self.api_key)

    def get_required_config(self) -> List[str]:
        return ["api_key"]

class AnthropicProvider(BaseSummaryProvider):
    """Anthropic Claude provider"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.model = config.get("model", "claude-3-haiku-20240307")
        self.base_url = config.get("base_url", "https://api.anthropic.com")

    async def generate_summary(self, request: SummaryRequest) -> SummaryResponse:
        import time
        start_time = time.time()

        try:
            prompt = self._build_prompt(request)

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/v1/messages",
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01"
                    },
                    json={
                        "model": self.model,
                        "max_tokens": request.max_length,
                        "temperature": request.temperature,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ]
                    }
                )

                if response.status_code != 200:
                    raise Exception(f"Anthropic API error: {response.status_code} - {response.text}")

                data = response.json()
                summary = data["content"][0]["text"].strip()

                return SummaryResponse(
                    summary=summary,
                    provider="Anthropic",
                    model=self.model,
                    tokens_used=data.get("usage", {}).get("input_tokens", 0) + data.get("usage", {}).get("output_tokens", 0),
                    processing_time=time.time() - start_time
                )

        except Exception as e:
            return SummaryResponse(
                summary="",
                provider="Anthropic",
                model=self.model,
                error=str(e),
                processing_time=time.time() - start_time
            )

    async def is_available(self) -> bool:
        return bool(self.api_key)

    def get_required_config(self) -> List[str]:
        return ["api_key"]

class OllamaProvider(BaseSummaryProvider):
    """Ollama local provider"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.model = config.get("model", "llama2")

    async def generate_summary(self, request: SummaryRequest) -> SummaryResponse:
        import time
        start_time = time.time()

        try:
            prompt = self._build_prompt(request)

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": request.temperature,
                            "num_predict": request.max_length
                        }
                    }
                )

                if response.status_code != 200:
                    raise Exception(f"Ollama API error: {response.status_code} - {response.text}")

                data = response.json()
                summary = data["response"].strip()

                return SummaryResponse(
                    summary=summary,
                    provider="Ollama",
                    model=self.model,
                    processing_time=time.time() - start_time
                )

        except Exception as e:
            return SummaryResponse(
                summary="",
                provider="Ollama",
                model=self.model,
                error=str(e),
                processing_time=time.time() - start_time
            )

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except:
            return False

    def get_required_config(self) -> List[str]:
        return []

class SummaryProviderFactory:
    """Factory for creating summary providers"""

    _providers = {
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.ANTHROPIC: AnthropicProvider,
        ProviderType.OLLAMA: OllamaProvider,
    }

    @classmethod
    def create_provider(cls, provider_type: ProviderType, config: Dict[str, Any]) -> BaseSummaryProvider:
        """Create a provider instance"""
        if provider_type not in cls._providers:
            raise ValueError(f"Unknown provider type: {provider_type}")

        provider_class = cls._providers[provider_type]
        return provider_class(config)

    @classmethod
    def get_available_providers(cls) -> List[ProviderType]:
        """Get list of available provider types"""
        return list(cls._providers.keys())

class MultiProviderSummaryEngine:
    """Engine that manages multiple summary providers with fallback support"""

    def __init__(self, provider_configs: Dict[str, Dict[str, Any]]):
        self.providers = {}
        self.fallback_order = []

        for provider_name, config in provider_configs.items():
            try:
                provider_type = ProviderType(provider_name.lower())
                provider = SummaryProviderFactory.create_provider(provider_type, config)
                self.providers[provider_name] = provider

                if config.get("enabled", True):
                    priority = config.get("priority", 50)
                    self.fallback_order.append((priority, provider_name))
            except (ValueError, Exception) as e:
                print(f"Failed to initialize provider {provider_name}: {e}")

        # Sort by priority (lower number = higher priority)
        self.fallback_order.sort(key=lambda x: x[0])

    async def generate_summary(self, request: SummaryRequest, preferred_provider: Optional[str] = None) -> SummaryResponse:
        """Generate summary with fallback support"""

        # Try preferred provider first
        if preferred_provider and preferred_provider in self.providers:
            provider = self.providers[preferred_provider]
            if await provider.is_available():
                response = await provider.generate_summary(request)
                if not response.error:
                    return response

        # Try providers in fallback order
        for _, provider_name in self.fallback_order:
            if provider_name == preferred_provider:
                continue  # Already tried

            provider = self.providers[provider_name]
            if await provider.is_available():
                response = await provider.generate_summary(request)
                if not response.error:
                    return response

        # All providers failed
        return SummaryResponse(
            summary="Summary generation failed: No available providers",
            provider="None",
            model="None",
            error="All configured providers failed or are unavailable"
        )

    async def get_provider_status(self) -> Dict[str, bool]:
        """Get availability status of all providers"""
        status = {}
        for name, provider in self.providers.items():
            status[name] = await provider.is_available()
        return status

    def list_providers(self) -> List[str]:
        """List all configured providers"""
        return list(self.providers.keys())
