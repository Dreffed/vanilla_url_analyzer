# mcp_troubleshoot.ps1
# Quick troubleshooting for MCP installation issues

Write-Host "🔧 MCP Installation Troubleshooter" -ForegroundColor Green
Write-Host "=" * 40 -ForegroundColor Yellow

Write-Host "`n🎯 QUICK FIX - Use Core Version (Recommended):" -ForegroundColor Cyan
Write-Host "pip install -r requirements_core.txt" -ForegroundColor White
Write-Host "python main_agnostic.py" -ForegroundColor White
Write-Host "# Works perfectly without MCP!" -ForegroundColor Green

Write-Host "`n🧪 Try Different MCP Versions:" -ForegroundColor Cyan
$versions = @("1.0.0", "0.9.0", "0.8.0", "0.7.0", "0.5.0", "0.4.0", "0.3.0")
foreach ($v in $versions) {
    Write-Host "pip install mcp==$v" -ForegroundColor Gray
}

Write-Host "`n🌐 Platform Features:" -ForegroundColor Yellow
Write-Host "✅ Multi-provider summaries" -ForegroundColor Green
Write-Host "✅ Web UI with provider selection" -ForegroundColor Green
Write-Host "✅ REST API with documentation" -ForegroundColor Green
Write-Host "✅ Batch processing" -ForegroundColor Green
Write-Host "✅ Automatic fallback" -ForegroundColor Green
Write-Host "❌ MCP server (only if MCP installs)" -ForegroundColor Red

Write-Host "`n💡 The core version is production-ready!" -ForegroundColor Green
