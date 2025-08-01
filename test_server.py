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
