import os
import requests
import io
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import pdfplumber

app = FastAPI()

DOCSTRANGE_API_KEY = os.environ.get("DOCSTRANGE_API_KEY")
if not DOCSTRANGE_API_KEY:
    raise ValueError("DOCSTRANGE_API_KEY environment variable not set.")

DOCSTRANGE_API_URL = "https://extraction-api.nanonets.com/extract"

def extract_with_pdfplumber(file_content: bytes, filename: str) -> str:
    """
    Fallback PDF extraction using pdfplumber
    Works for simple PDFs without complex layouts
    """
    try:
        print(f"[DocStrange] 📄 Using pdfplumber fallback for {filename}...")
        markdown_content = f"# {filename}\n\n"
        
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            total_pages = len(pdf.pages)
            print(f"[DocStrange] Total pages: {total_pages}")
            
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text
                text = page.extract_text()
                if text and text.strip():
                    markdown_content += f"\n\n## Sayfa {page_num}\n\n{text.strip()}"
                
                # Extract tables
                tables = page.extract_tables()
                if tables:
                    for table_idx, table in enumerate(tables, 1):
                        markdown_content += f"\n\n### Tablo {table_idx}\n\n"
                        # Convert table to markdown
                        if table:
                            # Header row
                            if len(table) > 0:
                                markdown_content += "| " + " | ".join(str(cell or "") for cell in table[0]) + " |\n"
                                markdown_content += "|" + "|".join(["---" for _ in table[0]]) + "|\n"
                            # Data rows
                            for row in table[1:]:
                                markdown_content += "| " + " | ".join(str(cell or "") for cell in row) + " |\n"
                        markdown_content += "\n"
                
                if page_num % 5 == 0:
                    print(f"[DocStrange] Processed {page_num}/{total_pages} pages")
        
        print(f"[DocStrange] ✅ pdfplumber extracted {len(markdown_content)} chars from {total_pages} pages")
        return markdown_content.strip()
        
    except Exception as e:
        print(f"[DocStrange] ❌ pdfplumber failed: {e}")
        raise HTTPException(status_code=500, detail=f"pdfplumber extraction failed: {e}")

@app.post("/convert/pdf-to-markdown")
async def convert_pdf_to_markdown(
    file: UploadFile = File(...),
    use_fallback: str = Form(default="false")
):
    """
    Converts a PDF file to Markdown.
    First tries Nanonets API (for complex/scanned PDFs with good OCR).
    Falls back to pdfplumber (for simple text PDFs) if Nanonets fails or is too slow.
    
    Parameters:
    - file: PDF file to convert
    - use_fallback: "true" to skip Nanonets and use pdfplumber directly, "false" to try Nanonets first
    """
    if not file.content_type == "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are accepted.")

    try:
        # Read file content once
        file_content = await file.read()
        markdown_content = None
        extraction_method = None
        use_fallback_bool = use_fallback.lower() == "true"
        
        # Option 1: Use pdfplumber directly if requested
        if use_fallback_bool:
            print("[DocStrange] Using pdfplumber (user requested)")
            markdown_content = extract_with_pdfplumber(file_content, file.filename)
            extraction_method = "pdfplumber"
        
        # Option 2: Try Nanonets first
        else:
            try:
                print(f"[DocStrange] 🌐 Trying Nanonets API for {file.filename}...")
                files_dict = {'file': (file.filename, file_content, file.content_type)}
                data = {'output_type': 'markdown'}
                headers = {'Authorization': f'Bearer {DOCSTRANGE_API_KEY}'}
                
                # Short timeout - if Nanonets is slow, use pdfplumber
                response = requests.post(
                    DOCSTRANGE_API_URL, 
                    headers=headers, 
                    files=files_dict, 
                    data=data, 
                    timeout=30  # 30 seconds max
                )
                
                if response.status_code == 200:
                    nanonets_response = response.json()
                    processing_status = nanonets_response.get('processing_status', 'completed')
                    
                    print(f"[DocStrange] Nanonets status: {processing_status}")
                    
                    # If async processing, use fallback immediately
                    if processing_status == 'processing':
                        print("[DocStrange] ⚠️ Nanonets returned async processing, using pdfplumber fallback")
                        markdown_content = extract_with_pdfplumber(file_content, file.filename)
                        extraction_method = "pdfplumber (Nanonets async)"
                    else:
                        # Try to extract content from Nanonets response
                        if 'content' in nanonets_response and nanonets_response['content']:
                            markdown_content = nanonets_response['content']
                            extraction_method = "nanonets"
                        elif 'data' in nanonets_response:
                            data_field = nanonets_response['data']
                            if isinstance(data_field, list) and len(data_field) > 0:
                                extracted = data_field[0]
                            else:
                                extracted = data_field
                            markdown_content = (
                                extracted.get('content', '') or
                                extracted.get('markdown', '') or 
                                extracted.get('text', '')
                            )
                            extraction_method = "nanonets"
                        
                        # If still no content, use fallback
                        if not markdown_content or not markdown_content.strip():
                            print("[DocStrange] ⚠️ Nanonets returned empty content, using pdfplumber fallback")
                            markdown_content = extract_with_pdfplumber(file_content, file.filename)
                            extraction_method = "pdfplumber (Nanonets empty)"
                else:
                    print(f"[DocStrange] ⚠️ Nanonets failed with status {response.status_code}, using pdfplumber fallback")
                    markdown_content = extract_with_pdfplumber(file_content, file.filename)
                    extraction_method = "pdfplumber (Nanonets error)"
                    
            except requests.Timeout:
                print("[DocStrange] ⏰ Nanonets timeout, using pdfplumber fallback")
                markdown_content = extract_with_pdfplumber(file_content, file.filename)
                extraction_method = "pdfplumber (Nanonets timeout)"
            except Exception as e:
                print(f"[DocStrange] ❌ Nanonets error: {e}, using pdfplumber fallback")
                markdown_content = extract_with_pdfplumber(file_content, file.filename)
                extraction_method = f"pdfplumber (Nanonets exception)"
        
        # Final check
        if not markdown_content or not markdown_content.strip():
            raise HTTPException(
                status_code=500,
                detail="Failed to extract any content from PDF using both Nanonets and pdfplumber"
            )
        
        # Return formatted response
        formatted_response = {
            "result": [
                {
                    "markdown": markdown_content.strip()
                }
            ],
            "extraction_method": extraction_method,
            "filename": file.filename
        }
        
        print(f"[DocStrange] ✅ Success with {extraction_method}")
        return JSONResponse(content=formatted_response)

    except HTTPException:
        raise
    except Exception as e:
        print(f"[DocStrange] ❌ Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.get("/health")
def health_check():
    return {"status": "ok", "methods": ["nanonets", "pdfplumber"]}
