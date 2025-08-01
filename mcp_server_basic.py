"""
Basic MCP Server for URL Research Platform v2.0
"""

import asyncio
import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, CallToolResult

# Import our main processor
try:
    from main_agnostic import URLProcessor, URLItem, SummaryConfig
except ImportError:
    print("Warning: Could not import main_agnostic. MCP server may not work correctly.")
    URLProcessor = None

class BasicMCPServer:
    def __init__(self):
        self.server = Server("url-research-v2")
        self.processor = URLProcessor() if URLProcessor else None
        self.setup_handlers()

    def setup_handlers(self):
        @self.server.list_tools()
        async def list_tools():
            from mcp.types import Tool
            return [
                Tool(
                    name="process_url",
                    description="Process a single URL and generate summary",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "provider": {"type": "string", "optional": True}
                        },
                        "required": ["url"]
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict):
            if name == "process_url" and self.processor:
                url = arguments.get("url")
                provider = arguments.get("provider")

                config = SummaryConfig(provider=provider) if provider else None
                url_item = URLItem(url=url)

                result = await self.processor.process_url(url_item, config)

                response_text = f"""# URL Analysis Results

**URL**: {result.url}
**Status**: {result.status}
**Title**: {result.title or 'N/A'}
**Provider**: {result.summary_provider or 'N/A'}
**Summary**: {result.generated_summary or result.error or 'N/A'}
"""

                return CallToolResult(
                    content=[TextContent(type="text", text=response_text)]
                )
            else:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Unknown tool: {name}")],
                    isError=True
                )

    async def run(self):
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream)

async def main():
    server = BasicMCPServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())
