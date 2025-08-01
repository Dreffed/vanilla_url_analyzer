"""
Configuration validator and management utility for multi-provider setup
"""

import os
import asyncio
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from summary_providers import ProviderType

class ValidationStatus(Enum):
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    MISSING = "missing"

@dataclass
class ValidationResult:
    status: ValidationStatus
    message: str
    suggestion: Optional[str] = None

class ConfigValidator:
    """Validates and manages provider configurations"""

    def __init__(self):
        self.required_env_vars = {
            ProviderType.OPENAI: ["OPENAI_API_KEY"],
            ProviderType.ANTHROPIC: ["ANTHROPIC_API_KEY"],
            ProviderType.OLLAMA: [],
        }

    def validate_provider_config(self, provider: ProviderType) -> Dict[str, ValidationResult]:
        """Validate configuration for a specific provider"""
        results = {}

        for var in self.required_env_vars.get(provider, []):
            value = os.getenv(var)

            if not value:
                results[var] = ValidationResult(
                    ValidationStatus.MISSING,
                    f"Required variable {var} is not set",
                    f"Set {var} in your .env file"
                )
            else:
                results[var] = ValidationResult(
                    ValidationStatus.VALID,
                    f"{var} is configured"
                )

        return results

    def validate_all_providers(self) -> Dict[ProviderType, Dict[str, ValidationResult]]:
        """Validate all provider configurations"""
        all_results = {}

        for provider in [ProviderType.OPENAI, ProviderType.ANTHROPIC, ProviderType.OLLAMA]:
            all_results[provider] = self.validate_provider_config(provider)

        return all_results

    def generate_config_report(self, validation_results: Dict[ProviderType, Dict[str, ValidationResult]]) -> str:
        """Generate a comprehensive configuration report"""
        report = []
        report.append("🔍 PROVIDER CONFIGURATION VALIDATION REPORT")
        report.append("=" * 60)

        for provider, results in validation_results.items():
            provider_name = provider.value.upper()
            report.append(f"\n📋 {provider_name}")

            if not results:
                report.append("   No configuration variables found")
                continue

            for var, result in results.items():
                if result.status == ValidationStatus.VALID:
                    emoji = "✅"
                elif result.status == ValidationStatus.WARNING:
                    emoji = "⚠️ "
                elif result.status == ValidationStatus.INVALID:
                    emoji = "❌"
                else:
                    emoji = "🔍"

                report.append(f"   {emoji} {var}: {result.message}")
                if result.suggestion:
                    report.append(f"      💡 {result.suggestion}")

        return "\n".join(report)

async def main():
    """Main function for configuration management"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        validator = ConfigValidator()

        if command == "validate":
            print("🔍 Validating provider configurations...")
            results = validator.validate_all_providers()
            report = validator.generate_config_report(results)
            print(report)
        else:
            print(f"Unknown command: {command}")
            print("Available commands: validate")
    else:
        print("🔧 URL Research Platform - Configuration Manager")
        print("Available commands: validate")

if __name__ == "__main__":
    asyncio.run(main())
