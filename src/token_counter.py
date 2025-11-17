"""
Token counting and analysis for different chunking strategies
"""

import tiktoken
from typing import List, Dict
from rich.console import Console
from rich.table import Table

console = Console()


class TokenCounter:
    """Count tokens using tiktoken (OpenAI's tokenizer)"""
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize tokenizer
        
        Args:
            encoding_name: Tokenizer to use
                - cl100k_base: GPT-4, GPT-3.5-turbo, text-embedding-ada-002
                - p50k_base: GPT-3 models (davinci, curie, etc.)
        """
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.encoding_name = encoding_name
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def estimate_cost(self, text: str, model: str = "claude-sonnet-4") -> Dict[str, float]:
        """
        Estimate API cost for processing this text
        
        Args:
            text: Input text
            model: Model name for pricing
        
        Returns:
            Dict with token count and estimated costs
        """
        tokens = self.count_tokens(text)
        
        # Claude Sonnet 4 pricing (per 1M tokens)
        pricing = {
            "claude-sonnet-4": {"input": 0.000003, "output": 0.000015},
            "claude-opus-4": {"input": 0.000015, "output": 0.000075},
            "claude-haiku": {"input": 0.00000025, "output": 0.00000125},
        }
        
        rates = pricing.get(model, pricing["claude-sonnet-4"])
        
        # Assume output is ~20% of input for extraction tasks
        estimated_output_tokens = int(tokens * 0.2)
        
        input_cost = tokens * rates["input"]
        output_cost = estimated_output_tokens * rates["output"]
        total_cost = input_cost + output_cost
        
        return {
            "input_tokens": tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "cost_per_1k_tokens": (total_cost / tokens) * 1000 if tokens > 0 else 0
        }
    
    def analyze_text(self, text: str, label: str = "Document") -> Dict:
        """Comprehensive token analysis"""
        
        tokens = self.count_tokens(text)
        characters = len(text)
        lines = len(text.split('\n'))
        words = len(text.split())
        
        # Calculate ratios
        chars_per_token = characters / tokens if tokens > 0 else 0
        tokens_per_word = tokens / words if words > 0 else 0
        
        result = {
            "label": label,
            "characters": characters,
            "lines": lines,
            "words": words,
            "tokens": tokens,
            "chars_per_token": chars_per_token,
            "tokens_per_word": tokens_per_word,
        }
        
        return result
    
    def display_analysis(self, analysis: Dict, show_costs: bool = True):
        """Display token analysis in a table"""
        
        table = Table(title=f"🔢 Token Analysis: {analysis['label']}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Characters", f"{analysis['characters']:,}")
        table.add_row("Lines", f"{analysis['lines']:,}")
        table.add_row("Words", f"{analysis['words']:,}")
        table.add_row("Tokens", f"{analysis['tokens']:,}")
        table.add_row("Chars/Token", f"{analysis['chars_per_token']:.2f}")
        table.add_row("Tokens/Word", f"{analysis['tokens_per_word']:.2f}")
        
        console.print(table)
        
        if show_costs:
            # Calculate costs
            text_sample = "x" * analysis['characters']  # Dummy text for cost calc
            costs = self.estimate_cost(text_sample)
            
            console.print("\n💰 Estimated API Costs (Claude Sonnet 4):\n")
            
            cost_table = Table()
            cost_table.add_column("Item", style="cyan")
            cost_table.add_column("Value", style="green")
            
            cost_table.add_row("Input Tokens", f"{costs['input_tokens']:,}")
            cost_table.add_row("Input Cost", f"${costs['input_cost']:.4f}")
            cost_table.add_row("Est. Output Tokens", f"{costs['estimated_output_tokens']:,}")
            cost_table.add_row("Est. Output Cost", f"${costs['output_cost']:.4f}")
            cost_table.add_row("Total Cost", f"${costs['total_cost']:.4f}")
            
            console.print(cost_table)


def main():
    """Test token counter"""
    
    # Sample texts
    samples = {
        "Short": "The borrower is TALF LLC.",
        "Medium": "The borrower is TALF LLC, a Delaware limited liability company. " * 10,
        "Long": "CREDIT AGREEMENT among TALF LLC, as Borrower, " * 100
    }
    
    counter = TokenCounter()
    
    for label, text in samples.items():
        console.print(f"\n{'='*60}\n")
        analysis = counter.analyze_text(text, label)
        counter.display_analysis(analysis, show_costs=(label == "Long"))


if __name__ == "__main__":
    main()
