"""
Simple tests for the URL Research Platform
"""

import asyncio
import os
from summary_providers import MultiProviderSummaryEngine, SummaryRequest

async def test_basic_functionality():
    """Test basic provider functionality"""
    print("🧪 Testing basic functionality...")

    # Mock configuration for testing
    test_configs = {
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY", "test-key"),
            "model": "gpt-3.5-turbo",
            "enabled": bool(os.getenv("OPENAI_API_KEY")),
            "priority": 10
        }
    }

    engine = MultiProviderSummaryEngine(test_configs)

    # Test provider status
    status = await engine.get_provider_status()
    print(f"✅ Provider status check complete: {status}")

    # Test summary generation (only if API keys are available)
    if any(status.values()):
        request = SummaryRequest(
            text_content="This is a test article about artificial intelligence.",
            title="AI Test Article",
            max_length=50
        )

        response = await engine.generate_summary(request)
        print(f"✅ Summary generation test: {response.provider} - {response.summary[:50]}...")
    else:
        print("⚠️  No providers available for summary test")

if __name__ == "__main__":
    asyncio.run(test_basic_functionality())
