import asyncio
import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from bs4 import BeautifulSoup
import json
import os
from datetime import datetime
from urllib.parse import urljoin, urlparse
from difflib import SequenceMatcher
from dotenv import load_dotenv

from summary_providers import (
    MultiProviderSummaryEngine,
    SummaryRequest,
    SummaryResponse,
    ProviderType
)

load_dotenv()

app = FastAPI(title="URL Research Platform", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLItem(BaseModel):
    url: str
    title: Optional[str] = None
    summary: Optional[str] = None
    usage: Optional[str] = None
    category: Optional[str] = None

class URLList(BaseModel):
    urls: List[URLItem]

class ProcessedResult(BaseModel):
    url: str
    status: str
    title: Optional[str] = None
    original_summary: Optional[str] = None
    generated_summary: Optional[str] = None
    summary_provider: Optional[str] = None
    summary_model: Optional[str] = None
    summary_tokens: Optional[int] = None
    accuracy_score: Optional[float] = None
    metadata: Dict[str, Any] = {}
    error: Optional[str] = None
    processing_time: float = 0

class SummaryConfig(BaseModel):
    provider: Optional[str] = None
    max_length: int = 150
    temperature: float = 0.3
    custom_prompt: Optional[str] = None

def load_provider_configs() -> Dict[str, Dict[str, Any]]:
    """Load provider configurations from environment variables"""
    configs = {}

    # OpenAI Configuration
    if os.getenv("OPENAI_API_KEY"):
        configs["openai"] = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "enabled": os.getenv("OPENAI_ENABLED", "true").lower() == "true",
            "priority": int(os.getenv("OPENAI_PRIORITY", "10"))
        }

    # Anthropic Configuration
    if os.getenv("ANTHROPIC_API_KEY"):
        configs["anthropic"] = {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
            "base_url": os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            "enabled": os.getenv("ANTHROPIC_ENABLED", "true").lower() == "true",
            "priority": int(os.getenv("ANTHROPIC_PRIORITY", "20"))
        }

    # Ollama Configuration
    ollama_enabled = os.getenv("OLLAMA_ENABLED", "false").lower() == "true"
    if ollama_enabled:
        configs["ollama"] = {
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "model": os.getenv("OLLAMA_MODEL", "llama2"),
            "enabled": True,
            "priority": int(os.getenv("OLLAMA_PRIORITY", "30"))
        }

    return configs

class URLProcessor:
    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )

        provider_configs = load_provider_configs()
        self.summary_engine = MultiProviderSummaryEngine(provider_configs)

    async def fetch_page(self, url: str) -> Dict[str, Any]:
        """Fetch page content and extract metadata"""
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            response = await self.client.get(url, follow_redirects=True)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            metadata = {
                'status_code': response.status_code,
                'final_url': str(response.url),
                'content_type': response.headers.get('content-type', ''),
                'content_length': len(response.text),
                'title': None,
                'description': None,
                'text_content': ''
            }

            # Extract title
            title_tag = soup.find('title')
            if title_tag:
                metadata['title'] = title_tag.get_text().strip()

            # Extract text content
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            metadata['text_content'] = text[:5000]

            return metadata

        except httpx.HTTPStatusError as e:
            return {'error': f'HTTP {e.response.status_code}: {e.response.reason_phrase}'}
        except Exception as e:
            return {'error': str(e)}

    async def generate_summary(self, text_content: str, title: str = "", config: SummaryConfig = None) -> SummaryResponse:
        """Generate summary using the multi-provider engine"""
        if not config:
            config = SummaryConfig()

        request = SummaryRequest(
            text_content=text_content,
            title=title,
            max_length=config.max_length,
            temperature=config.temperature,
            custom_prompt=config.custom_prompt
        )

        return await self.summary_engine.generate_summary(request, config.provider)

    def calculate_accuracy_score(self, original: str, generated: str) -> float:
        """Calculate similarity score between original and generated summaries"""
        if not original or not generated:
            return 0.0

        similarity = SequenceMatcher(None, original.lower(), generated.lower()).ratio()
        return round(similarity * 100, 2)

    async def process_url(self, url_item: URLItem, summary_config: SummaryConfig = None) -> ProcessedResult:
        """Process a single URL"""
        start_time = datetime.now()

        try:
            metadata = await self.fetch_page(url_item.url)

            if 'error' in metadata:
                return ProcessedResult(
                    url=url_item.url,
                    status="error",
                    error=metadata['error'],
                    processing_time=(datetime.now() - start_time).total_seconds()
                )

            summary_response = await self.generate_summary(
                metadata['text_content'],
                metadata.get('title', ''),
                summary_config
            )

            accuracy_score = None
            if url_item.summary and not summary_response.error:
                accuracy_score = self.calculate_accuracy_score(
                    url_item.summary,
                    summary_response.summary
                )

            return ProcessedResult(
                url=url_item.url,
                status="success" if not summary_response.error else "error",
                title=metadata.get('title'),
                original_summary=url_item.summary,
                generated_summary=summary_response.summary if not summary_response.error else None,
                summary_provider=summary_response.provider,
                summary_model=summary_response.model,
                summary_tokens=summary_response.tokens_used,
                accuracy_score=accuracy_score,
                metadata=metadata,
                error=summary_response.error,
                processing_time=(datetime.now() - start_time).total_seconds()
            )

        except Exception as e:
            return ProcessedResult(
                url=url_item.url,
                status="error",
                error=str(e),
                processing_time=(datetime.now() - start_time).total_seconds()
            )

    async def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all summary providers"""
        status = await self.summary_engine.get_provider_status()
        return {
            "available_providers": list(status.keys()),
            "provider_status": status,
            "total_providers": len(status),
            "available_count": sum(1 for available in status.values() if available)
        }

    async def close(self):
        await self.client.aclose()

processor = URLProcessor()

@app.on_event("shutdown")
async def shutdown_event():
    await processor.close()

@app.get("/providers/status")
async def get_provider_status():
    """Get status of all configured summary providers"""
    return await processor.get_provider_status()

@app.post("/process-urls")
async def process_urls(url_list: URLList, summary_config: SummaryConfig = None):
    """Process a list of URLs with configurable summary generation"""
    results = []

    semaphore = asyncio.Semaphore(5)

    async def process_with_semaphore(url_item):
        async with semaphore:
            return await processor.process_url(url_item, summary_config)

    tasks = [process_with_semaphore(url_item) for url_item in url_list.urls]
    results = await asyncio.gather(*tasks)

    return {
        "results": results,
        "summary": {
            "total": len(results),
            "successful": len([r for r in results if r.status == "success"]),
            "errors": len([r for r in results if r.status == "error"]),
        }
    }

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the enhanced UI"""
    html_content = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>URL Research Platform v2.0</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h1 { color: #333; margin-bottom: 30px; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            textarea, input, select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
            textarea { height: 200px; resize: vertical; }
            button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
            button:hover { background-color: #0056b3; }
            .loading { display: none; margin: 20px 0; text-align: center; }
            .results { margin-top: 30px; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 URL Research Platform v2.0 - Multi-Provider</h1>

            <div class="form-group">
                <label for="urlInput">Enter URLs (JSON format or one URL per line):</label>
                <textarea id="urlInput" placeholder='[{"url": "https://example.com", "title": "Example"}]'></textarea>
            </div>

            <div class="form-group">
                <label for="providerSelect">Summary Provider:</label>
                <select id="providerSelect">
                    <option value="">Auto (fallback)</option>
                    <option value="openai">OpenAI</option>
                    <option value="anthropic">Anthropic</option>
                    <option value="ollama">Ollama</option>
                </select>
            </div>

            <button onclick="processUrls()" id="processBtn">Process URLs</button>

            <div class="loading" id="loading">
                <p>Processing URLs...</p>
            </div>

            <div id="message"></div>
            <div id="results" class="results"></div>
        </div>

        <script>
            async function processUrls() {
                const urlInput = document.getElementById('urlInput').value.trim();
                const provider = document.getElementById('providerSelect').value;
                const messageDiv = document.getElementById('message');
                const resultsDiv = document.getElementById('results');
                const loadingDiv = document.getElementById('loading');

                if (!urlInput) {
                    messageDiv.innerHTML = '<div style="color: red;">Please enter URLs</div>';
                    return;
                }

                loadingDiv.style.display = 'block';
                messageDiv.innerHTML = '';
                resultsDiv.innerHTML = '';

                try {
                    let urls;
                    try {
                        urls = JSON.parse(urlInput);
                    } catch {
                        urls = urlInput.split('\\n').map(url => ({url: url.trim()})).filter(u => u.url);
                    }

                    const requestBody = { urls };
                    if (provider) requestBody.provider = provider;

                    const response = await fetch('/process-urls', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(requestBody)
                    });

                    const data = await response.json();
                    displayResults(data);

                } catch (error) {
                    messageDiv.innerHTML = `<div style="color: red;">Error: ${error.message}</div>`;
                } finally {
                    loadingDiv.style.display = 'none';
                }
            }

            function displayResults(data) {
                const resultsDiv = document.getElementById('results');

                let html = '<h2>Results</h2><table><thead><tr>';
                html += '<th>URL</th><th>Status</th><th>Title</th><th>Summary</th><th>Provider</th>';
                html += '</tr></thead><tbody>';

                data.results.forEach(result => {
                    html += '<tr>';
                    html += `<td><a href="${result.url}" target="_blank">${result.url}</a></td>`;
                    html += `<td>${result.status}</td>`;
                    html += `<td>${result.title || '-'}</td>`;
                    html += `<td>${result.generated_summary || result.error || '-'}</td>`;
                    html += `<td>${result.summary_provider || '-'}</td>`;
                    html += '</tr>';
                });

                html += '</tbody></table>';
                html += `<p>Summary: ${data.summary.successful}/${data.summary.total} successful</p>`;

                resultsDiv.innerHTML = html;
            }
        </script>
    </body>
    </html>
    '''
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
