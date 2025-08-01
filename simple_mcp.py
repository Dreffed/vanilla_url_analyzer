import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

class MCPServer:
    """
    A simple MCP (Metadata Content Processor) server.

    Accepts a list of URLs, fetches metadata, and generates a report.
    """

    def __init__(self):
        self.report = []

    def process_urls(self, url_list_str):
        """
        Processes a list of URLs provided as a string (CSV, JSON, or list of URLs).

        Args:
            url_list_str:  A string containing a list of URLs, CSV format, or JSON.
        """

        try:
            # Attempt to parse as JSON
            url_list = json.loads(url_list_str)
            print(f"JSON: \n{url_list}\n{=*5}")
        except json.JSONDecodeError:
            # If not JSON, attempt to parse as CSV
            try:
                url_list = [line.strip() for line in url_list_str.splitlines()]
            except:
                raise ValueError("Invalid input format.  Must be JSON or CSV string.")

        for item in url_list:
            if isinstance(item, str):
                url = item.strip()
            else:
                url = item.get('url')

            if not url.strip():  # Skip empty strings
                continue

            try:
                self.fetch_metadata(url)
            except Exception as e:
                print(f"Error processing {url}: {e}")  # Log errors, but don't halt processing

    def fetch_metadata(self, url):
        """
        Fetches the content of a URL, extracts metadata, and creates a summary.

        Args:
            url: The URL to fetch.
        """
        try:
            response = requests.get(url, timeout=10)  #Added timeout for robustness
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            soup = BeautifulSoup(response.content, 'html.parser')
            title = soup.title.string if soup.title else "No Title"
            description = soup.find("meta", attrs={"name": "description"})["content"] if soup.find("meta", attrs={"name": "description"}) else ""
            # Add other metadata extraction logic here -  e.g.,  h1, h2, etc.
            # For example to find the summary:
            summary_meta = soup.find("meta", attrs={"name": "summary"})
            summary = summary_meta["content"] if summary_meta else ""

            metadata = {
                "url": url,
                "title": title,
                "description": description,
                "summary": summary
            }

            self.report.append(metadata)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            self.report.append({"url": url, "error": str(e)})
        except Exception as e:
            print(f"Error processing {url}: {e}")
            self.report.append({"url": url, "error": str(e)})


    def generate_report(self):
        """
        Generates a JSON report of the processed URLs and their metadata.
        """
        return json.dumps(self.report, indent=4) #Improved output for readability


# Example Usage (Simulated Server Interaction)
if __name__ == '__main__':
    server = MCPServer()

    # Example input - JSON format
    json_url_list = '[{"url": "https://www.example.com"}, {"url": "https://www.python.org"}]'
    server.process_urls(json_url_list)
    print("JSON Report:")
    print(server.generate_report())


    # Example input - CSV format
    csv_url_list = "https://www.example.com,https://www.python.org"
    server.process_urls(csv_url_list)
    print("\nCSV Report:")
    print(server.generate_report())

    # Example error handling
    bad_url_list = "https://www.example.com,https://invalid.example.com"
    server.process_urls(bad_url_list)
    print("\nReport with error:")
    print(server.generate_report())
