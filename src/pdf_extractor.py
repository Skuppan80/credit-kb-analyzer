"""
PDF text extraction with metadata
"""

import pdfplumber
from pathlib import Path
from typing import Dict, Any
from rich.console import Console
from rich.table import Table

console = Console()


class PDFExtractor:
    """Extract text and metadata from PDF files"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    def extract_text(self) -> str:
        """Extract all text from PDF"""
        console.print(f"\n📄 Extracting text from: {self.pdf_path.name}")
        
        full_text = []
        
        with pdfplumber.open(self.pdf_path) as pdf:
            total_pages = len(pdf.pages)
            console.print(f"   Pages: {total_pages}")
            
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    full_text.append(text)
                
                if i % 10 == 0:
                    console.print(f"   Processed: {i}/{total_pages} pages...")
        
        combined_text = "\n\n".join(full_text)
        console.print(f"✅ Extracted {len(combined_text):,} characters")
        
        return combined_text
    
    def extract_with_metadata(self) -> Dict[str, Any]:
        """Extract text with detailed metadata"""
        console.print(f"\n📊 Analyzing PDF: {self.pdf_path.name}\n")
        
        result = {
            'filename': self.pdf_path.name,
            'text': '',
            'metadata': {},
            'pages_info': []
        }
        
        with pdfplumber.open(self.pdf_path) as pdf:
            # Overall metadata
            result['metadata'] = {
                'total_pages': len(pdf.pages),
                'file_size_mb': self.pdf_path.stat().st_size / (1024 * 1024),
                'pdf_metadata': pdf.metadata or {}
            }
            
            # Extract text and analyze each page
            all_text = []
            
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                all_text.append(text)
                
                # Page-level info
                page_info = {
                    'page_number': i,
                    'char_count': len(text),
                    'line_count': len(text.split('\n')),
                    'has_tables': len(page.extract_tables()) > 0
                }
                result['pages_info'].append(page_info)
            
            result['text'] = "\n\n".join(all_text)
            result['metadata']['total_characters'] = len(result['text'])
        
        # Display summary
        self._display_summary(result)
        
        return result
    
    def _display_summary(self, result: Dict[str, Any]):
        """Display extraction summary in a nice table"""
        
        table = Table(title=f"📄 PDF Analysis: {result['filename']}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        meta = result['metadata']
        
        table.add_row("Total Pages", str(meta['total_pages']))
        table.add_row("File Size", f"{meta['file_size_mb']:.2f} MB")
        table.add_row("Total Characters", f"{meta['total_characters']:,}")
        table.add_row("Pages with Tables", 
                     str(sum(1 for p in result['pages_info'] if p['has_tables'])))
        
        console.print(table)


def main():
    """Test the extractor"""
    from pathlib import Path
    
    # Find PDF in data/pdfs
    pdf_dir = Path("data/pdfs")
    pdfs = list(pdf_dir.glob("*.pdf"))
    
    if not pdfs:
        console.print("❌ No PDFs found in data/pdfs/")
        console.print("   Add a PDF file to data/pdfs/ first")
        return
    
    # Use first PDF found
    pdf_path = pdfs[0]
    console.print(f"🔍 Found PDF: {pdf_path.name}\n")
    
    # Extract with metadata
    extractor = PDFExtractor(pdf_path)
    result = extractor.extract_with_metadata()
    
    # Show first 500 characters
    console.print("\n📝 First 500 characters:\n")
    console.print(result['text'][:500])
    console.print("...\n")


if __name__ == "__main__":
    main()
