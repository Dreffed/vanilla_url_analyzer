#!/usr/bin/env python3
"""
Startup script for URL Research Platform v2.0
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_requirements():
    """Check if requirements are installed"""
    try:
        import fastapi
        import httpx
        import beautifulsoup4
        print("✅ Core dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install -r requirements_updated.txt")
        return False

def check_env_file():
    """Check if .env file exists"""
    if not Path(".env").exists():
        print("📝 No .env file found")
        if Path(".env.example").exists():
            print("Copying .env.example to .env...")
            import shutil
            shutil.copy(".env.example", ".env")
            print("✅ Created .env file - please configure your API keys")
            return False
        else:
            print("❌ No .env.example found")
            return False
    print("✅ .env file found")
    return True

async def validate_config():
    """Validate provider configuration"""
    try:
        from config_validator import ConfigValidator
        validator = ConfigValidator()
        results = validator.validate_all_providers()

        total_valid = sum(
            1 for provider_results in results.values()
            for result in provider_results.values()
            if result.status.value == "valid"
        )

        print(f"✅ Configuration validation complete ({total_valid} valid settings)")
        return True
    except Exception as e:
        print(f"⚠️  Configuration validation failed: {e}")
        return False

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting URL Research Platform v2.0...")
    print("🌐 Web UI will be available at: http://localhost:8000")
    print("📚 API docs will be available at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    print("-" * 60)

    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "main_agnostic:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n👋 Server stopped")

async def main():
    """Main startup function"""
    print("🚀 URL Research Platform v2.0 - Startup Check")
    print("=" * 60)

    # Pre-flight checks
    if not check_python_version():
        return

    if not check_requirements():
        return

    env_ok = check_env_file()

    if env_ok:
        await validate_config()

    if env_ok:
        start_server()
    else:
        print("\n💡 Next steps:")
        print("1. Edit .env file with your API keys")
        print("2. Run: python startup.py")

if __name__ == "__main__":
    asyncio.run(main())
