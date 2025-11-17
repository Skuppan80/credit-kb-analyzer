"""
Analyze credit PDF: Extract text and count tokens
"""

from pathlib import Path
from src.pdf_extractor import PDFExtractor
from src.token_counter import TokenCounter
from rich.console import Console
import json

console = Console()


def analyze_credit_pdf(pdf_path: str):
    """Complete analysis of credit PDF"""
    
    console.print("\n" + "="*60)
    console.print("📊 CREDIT DOCUMENT ANALYSIS")
    console.print("="*60 + "\n")
    
    # Step 1: Extract text
    extractor = PDFExtractor(pdf_path)
    result = extractor.extract_with_metadata()
    
    # Step 2: Analyze tokens
    console.print("\n" + "="*60)
    console.print("🔢 TOKEN ANALYSIS")
    console.print("="*60 + "\n")
    
    counter = TokenCounter()
    analysis = counter.analyze_text(result['text'], result['filename'])
    counter.display_analysis(analysis, show_costs=True)
    
    # Step 3: Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Save extracted text
    text_file = output_dir / f"{Path(pdf_path).stem}_extracted.txt"
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(result['text'])
    
    console.print(f"\n💾 Saved extracted text to: {text_file}")
    
    # Save analysis
    analysis_file = output_dir / f"{Path(pdf_path).stem}_analysis.json"
    
    analysis_data = {
        "filename": result['filename'],
        "metadata": result['metadata'],
        "token_analysis": analysis,
        "extraction_timestamp": str(Path(pdf_path).stat().st_mtime)
    }
    
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2)
    
    console.print(f"💾 Saved analysis to: {analysis_file}")
    
    # Display summary
    console.print("\n" + "="*60)
    console.print("📋 SUMMARY")
    console.print("="*60 + "\n")
    
    console.print(f"Document: {result['filename']}")
    console.print(f"Total tokens: {analysis['tokens']:,}")
    console.print(f"Characters: {analysis['characters']:,}")
    console.print(f"Ratio: {analysis['chars_per_token']:.2f} chars/token")
    console.print(f"\n✅ Ready for chunking experiments!\n")
    
    return result, analysis


def main():
    """Main entry point"""
    
    # Find PDF
    pdf_dir = Path("data/pdfs")
    pdfs = list(pdf_dir.glob("*.pdf"))
    
    if not pdfs:
        console.print("\n❌ No PDFs found in data/pdfs/\n")
        console.print("Please add a credit agreement PDF to data/pdfs/\n")
        return
    
    # Analyze first PDF
    pdf_path = pdfs[0]
    analyze_credit_pdf(str(pdf_path))


if __name__ == "__main__":
    main()
