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
