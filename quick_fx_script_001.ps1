# quick_fix.ps1
# Quick fix script for the forEach error

Write-Host "🔧 Quick Fix for URL Research Platform" -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Yellow

Write-Host "`n🎯 Applying fixes for the forEach error..." -ForegroundColor Cyan

# Create a simple working test server
$simpleTestServer = @'
"""
Simple test server to debug the forEach issue
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import json

app = FastAPI()

@app.get("/")
async def root():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head><title>Debug Test</title></head>
    <body>
        <h1>Debug Test</h1>
        <button onclick="testSimple()">Test Simple</button>
        <div id="result"></div>

        <script>
            async function testSimple() {
                try {
                    const response = await fetch('/test');
                    const data = await response.json();

                    console.log('Data received:', data);

                    if (data.results && Array.isArray(data.results)) {
                        document.getElementById('result').innerHTML =
                            '<p>✅ Success! Found ' + data.results.length + ' results</p>';
                    } else {
                        document.getElementById('result').innerHTML =
                            '<p>❌ No results array found</p>';
                    }
                } catch (error) {
                    document.getElementById('result').innerHTML =
                        '<p style="color: red;">Error: ' + error.message + '</p>';
                }
            }
        </script>
    </body>
    </html>
    """)

@app.get("/test")
async def test():
    return {
        "results": [
            {"url": "test1", "status": "success", "title": "Test 1"},
            {"url": "test2", "status": "success", "title": "Test 2"}
        ],
        "summary": {"total": 2, "successful": 2, "errors": 0}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'@

$simpleTestServer | Out-File -FilePath "test_server.py" -Encoding UTF8
Write-Host "✅ Created test_server.py" -ForegroundColor Green

# Create a minimal .env for testing
$minimalEnv = @"
# Minimal configuration for testing
OPENAI_API_KEY=sk-test-key-12345-not-real
OPENAI_ENABLED=false
OPENAI_PRIORITY=10
"@

if (!(Test-Path ".env")) {
    $minimalEnv | Out-File -FilePath ".env" -Encoding UTF8
    Write-Host "✅ Created minimal .env file" -ForegroundColor Green
} else {
    Write-Host "⚠️  .env file already exists" -ForegroundColor Yellow
}

# Create debug commands
$debugCommands = @"
# Debug Commands for URL Research Platform

## Step 1: Test basic FastAPI server
python test_server.py
# Open: http://localhost:8000
# Click "Test Simple" - should show "Success! Found 2 results"

## Step 2: Test main application
python main_agnostic.py
# Open: http://localhost:8000/debug/test
# Should return JSON with status "ok"

## Step 3: Test provider status
# Open: http://localhost:8000/providers/status
# Should return provider configuration

## Step 4: Test with browser console
# 1. Open browser DevTools (F12)
# 2. Go to Console tab
# 3. Try entering a URL in the main app
# 4. Look for JavaScript errors

## Common Issues and Fixes:

### Issue 1: "Cannot read properties of undefined (reading 'forEach')"
Fix: The API is not returning the expected data structure
Solution: Check that /process-urls returns { results: [...] }

### Issue 2: Provider not configured
Fix: No API keys in .env file
Solution: Add real API keys or use test mode

### Issue 3: CORS errors
Fix: Frontend and backend on different domains
Solution: Both should be on localhost:8000

### Issue 4: Network errors
Fix: Backend not running or wrong port
Solution: Ensure uvicorn is running on port 8000

## Debug URLs:
http://localhost:8000/debug/test - Basic API test
http://localhost:8000/providers/status - Provider status
http://localhost:8000/debug/config - Configuration check

## Test with curl:
curl -X POST http://localhost:8000/process-urls -H "Content-Type: application/json" -d '{"urls":[{"url":"https://httpbin.org/json"}]}'
"@

$debugCommands | Out-File -FilePath "DEBUG_COMMANDS.md" -Encoding UTF8
Write-Host "✅ Created DEBUG_COMMANDS.md" -ForegroundColor Green

Write-Host "`n🚀 Quick Fix Applied!" -ForegroundColor Green
Write-Host "=" * 30 -ForegroundColor Yellow

Write-Host "`n📋 Next Steps:" -ForegroundColor Cyan
Write-Host "1. First test the simple server:" -ForegroundColor White
Write-Host "   python test_server.py" -ForegroundColor Gray
Write-Host "   Open: http://localhost:8000" -ForegroundColor Gray

Write-Host "`n2. If that works, test the main app:" -ForegroundColor White
Write-Host "   python main_agnostic.py" -ForegroundColor Gray

Write-Host "`n3. Debug specific issue:" -ForegroundColor White
Write-Host "   Open browser DevTools (F12) and check Console tab" -ForegroundColor Gray

Write-Host "`n4. Check the debug guide:" -ForegroundColor White
Write-Host "   Read DEBUG_COMMANDS.md for detailed troubleshooting" -ForegroundColor Gray

Write-Host "`n💡 Most likely causes:" -ForegroundColor Yellow
Write-Host "- API returning wrong data structure (not { results: [...] })" -ForegroundColor Cyan
Write-Host "- No providers configured (missing API keys)" -ForegroundColor Cyan
Write-Host "- Network/CORS issues" -ForegroundColor Cyan

Write-Host "`n🎯 The test_server.py will help isolate the issue!" -ForegroundColor Green