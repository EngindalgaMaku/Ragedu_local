import requests
import json
import os

BASE_URL = "http://127.0.0.1:8001"
PROCESS_ENDPOINT = "/process"
PDF_PATH = "test_documents/test_document.pdf"

def test_pdf_conversion():
    """Tests the PDF to Markdown conversion endpoint."""
    print("--- Starting PDF Conversion Test ---")

    if not os.path.exists(PDF_PATH):
        print(f"❌ FAILED: Test PDF not found at '{PDF_PATH}'")
        return

    try:
        url = BASE_URL + PROCESS_ENDPOINT
        print(f"Testing endpoint: {url}")
        
        with open(PDF_PATH, "rb") as f:
            files = {"file": (os.path.basename(PDF_PATH), f, "application/pdf")}
            response = requests.post(url, files=files, timeout=300)

        if response.status_code == 200:
            print("✅ Status Code: 200")
            try:
                data = response.json()
                print("Response JSON (metadata only):")
                # Print metadata, but not the full content
                metadata = data.get("metadata", {})
                print(json.dumps(metadata, indent=2))

                content = data.get("content", "")
                if content and len(content) > 100:
                    print(f"✅ Successfully received markdown content (length: {len(content)}).")
                    # Save the markdown for inspection
                    output_filename = "test_output.md"
                    with open(output_filename, "w", encoding="utf-8") as md_file:
                        md_file.write(content)
                    print(f"✅ Markdown content saved to '{output_filename}'")
                else:
                    print("❌ FAILED: Markdown content is missing or too short.")
            except json.JSONDecodeError:
                print("❌ FAILED: Failed to decode JSON response.")
        else:
            print(f"❌ FAILED: Endpoint returned status code {response.status_code}")
            print(f"Response text: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"❌ FAILED: Could not connect to {url}. Error: {e}")

    print("\n--- PDF Conversion Test Complete ---")

if __name__ == "__main__":
    test_pdf_conversion()