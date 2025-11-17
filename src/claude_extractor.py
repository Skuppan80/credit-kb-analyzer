"""
Extract credit terms using Claude API from retrieved chunks
"""

from typing import List, Dict, Any
import anthropic
import json
import os
from dotenv import load_dotenv
from rich.console import Console
from .token_counter import TokenCounter

load_dotenv()
console = Console()


class ClaudeExtractor:
    """
    Extract structured data from text chunks using Claude API
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize Claude API client
        
        Args:
            api_key: Anthropic API key (or uses ANTHROPIC_API_KEY env var)
        """
        api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        
        if not api_key or api_key == 'your-api-key-here':
            raise ValueError("ANTHROPIC_API_KEY not set. Add it to .env file")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-5-20250929"
        self.token_counter = TokenCounter()
        
        console.print(f"\n🤖 Claude API initialized")
        console.print(f"   Model: {self.model}")
    
    def extract_from_chunks(self, chunks: List[Dict[str, Any]], 
                           prompt_template: str = None) -> Dict[str, Any]:
        """
        Extract credit terms from retrieved chunks
        
        Args:
            chunks: List of retrieved chunks
            prompt_template: Custom prompt (uses default if None)
            
        Returns:
            Extracted data + usage stats
        """
        # Combine chunks into context
        context = self._combine_chunks(chunks)
        
        # Count input tokens
        input_tokens = self.token_counter.count_tokens(context)
        
        console.print(f"\n📊 Extraction Info:")
        console.print(f"   Chunks: {len(chunks)}")
        console.print(f"   Input tokens: {input_tokens:,}")
        
        # Build prompt
        if prompt_template is None:
            prompt = self._build_default_prompt(context)
        else:
            prompt = prompt_template.format(context=context)
        
        # Call Claude API
        console.print(f"   Calling Claude API...")
        
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Parse response
            response_text = message.content[0].text
            
            # Extract JSON from response
            extraction = self._parse_json_response(response_text)
            
            # Calculate costs
            input_cost = message.usage.input_tokens * 0.000003
            output_cost = message.usage.output_tokens * 0.000015
            total_cost = input_cost + output_cost
            
            console.print(f"   ✅ Extraction complete")
            console.print(f"   Output tokens: {message.usage.output_tokens:,}")
            console.print(f"   Cost: ${total_cost:.4f}")
            
            return {
                "extraction": extraction,
                "usage": {
                    "input_tokens": message.usage.input_tokens,
                    "output_tokens": message.usage.output_tokens,
                    "input_cost": input_cost,
                    "output_cost": output_cost,
                    "total_cost": total_cost,
                },
                "num_chunks_used": len(chunks),
                "stop_reason": message.stop_reason
            }
            
        except Exception as e:
            console.print(f"   ❌ Error: {e}")
            return {
                "extraction": {"error": str(e)},
                "usage": {},
                "num_chunks_used": len(chunks)
            }
    
    def extract_from_full_document(self, full_text: str) -> Dict[str, Any]:
        """
        Extract from full document (baseline for comparison)
        
        Args:
            full_text: Complete document text
            
        Returns:
            Extracted data + usage stats
        """
        console.print(f"\n📊 Full Document Extraction:")
        
        # Count tokens
        input_tokens = self.token_counter.count_tokens(full_text)
        console.print(f"   Input tokens: {input_tokens:,}")
        
        # Truncate if needed
        max_chars = 400000  # ~100K tokens
        if len(full_text) > max_chars:
            console.print(f"   ⚠️  Truncating to {max_chars:,} chars")
            full_text = full_text[:max_chars]
        
        # Build prompt
        prompt = self._build_default_prompt(full_text)
        
        # Call API
        console.print(f"   Calling Claude API...")
        
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            response_text = message.content[0].text
            extraction = self._parse_json_response(response_text)
            
            input_cost = message.usage.input_tokens * 0.000003
            output_cost = message.usage.output_tokens * 0.000015
            total_cost = input_cost + output_cost
            
            console.print(f"   ✅ Extraction complete")
            console.print(f"   Output tokens: {message.usage.output_tokens:,}")
            console.print(f"   Cost: ${total_cost:.4f}")
            
            return {
                "extraction": extraction,
                "usage": {
                    "input_tokens": message.usage.input_tokens,
                    "output_tokens": message.usage.output_tokens,
                    "input_cost": input_cost,
                    "output_cost": output_cost,
                    "total_cost": total_cost,
                },
                "stop_reason": message.stop_reason
            }
            
        except Exception as e:
            console.print(f"   ❌ Error: {e}")
            return {
                "extraction": {"error": str(e)},
                "usage": {}
            }
    
    def _combine_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Combine chunks into a single context"""
        texts = []
        for i, chunk in enumerate(chunks, 1):
            texts.append(f"[Chunk {i}]\n{chunk['text']}")
        
        return "\n\n".join(texts)
    
    def _build_default_prompt(self, context: str) -> str:
        """Build default extraction prompt"""
        return f"""Extract credit agreement terms from the following document chunks and return as JSON.

Extract these fields:
- borrower: {{name, entity_type, jurisdiction}}
- lender: [{{name, role, commitment}}] (array if multiple)
- loan_details: {{total_amount, facility_type, purpose, currency}}
- interest_terms: {{base_rate, margin, total_rate, payment_frequency}}
- maturity: {{effective_date, maturity_date, term}}
- fees: {{origination_fee, commitment_fee}}
- financial_covenants: [array of covenant descriptions]
- collateral: [array of collateral descriptions]

Document chunks:

{context}

Return ONLY valid JSON with the extracted information. If a field is not found, use null.
"""
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON from Claude's response"""
        import re
        
        # Try to find JSON in response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                return {"error": "Failed to parse JSON", "raw_response": response_text}
        else:
            return {"error": "No JSON found in response", "raw_response": response_text}


def main():
    """Test Claude extractor"""
    
    console.print("\n" + "="*70)
    console.print("🧪 TESTING CLAUDE EXTRACTOR")
    console.print("="*70)
    
    # Test with sample chunks
    sample_chunks = [
        {
            "text": "The borrower is TALF LLC, a Delaware limited liability company.",
            "chunk_id": "chunk_0",
            "metadata": {"token_count": 15}
        },
        {
            "text": "The lender is the Federal Reserve Bank of New York, acting as Senior Lender with a commitment of $180,000,000,000.",
            "chunk_id": "chunk_1",
            "metadata": {"token_count": 25}
        },
        {
            "text": "The total facility amount is $200,000,000,000. The facility type is Term Asset-Backed Securities Loan Facility (TALF).",
            "chunk_id": "chunk_2",
            "metadata": {"token_count": 30}
        },
    ]
    
    try:
        extractor = ClaudeExtractor()
        
        # Test extraction
        result = extractor.extract_from_chunks(sample_chunks)
        
        console.print(f"\n📄 Extracted Data:")
        console.print(json.dumps(result['extraction'], indent=2))
        
        console.print(f"\n💰 Cost Analysis:")
        usage = result['usage']
        console.print(f"   Input tokens: {usage.get('input_tokens', 0):,}")
        console.print(f"   Output tokens: {usage.get('output_tokens', 0):,}")
        console.print(f"   Total cost: ${usage.get('total_cost', 0):.4f}")
        
    except ValueError as e:
        console.print(f"\n❌ {e}")
        console.print("   Please add your API key to .env file")


if __name__ == "__main__":
    main()
