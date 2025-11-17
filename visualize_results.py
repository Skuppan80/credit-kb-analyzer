"""
Visualize RAG evaluation results
"""

from pathlib import Path
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.console import Console

console = Console()


def load_results():
    """Load evaluation results"""
    results_file = Path("results/rag_evaluation_results.json")
    
    if not results_file.exists():
        console.print("❌ No results found. Run evaluate_rag.py first.")
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)


def create_cost_comparison_chart(results):
    """Create cost comparison bar chart"""
    
    strategies = []
    input_costs = []
    output_costs = []
    total_costs = []
    
    for key, data in results.items():
        usage = data['usage']
        strategies.append(data['strategy'])
        input_costs.append(usage.get('input_cost', 0))
        output_costs.append(usage.get('output_cost', 0))
        total_costs.append(usage.get('total_cost', 0))
    
    # Create stacked bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Input Cost',
        x=strategies,
        y=input_costs,
        marker_color='lightblue',
        text=[f'${c:.4f}' for c in input_costs],
        textposition='inside'
    ))
    
    fig.add_trace(go.Bar(
        name='Output Cost',
        x=strategies,
        y=output_costs,
        marker_color='lightcoral',
        text=[f'${c:.4f}' for c in output_costs],
        textposition='inside'
    ))
    
    fig.update_layout(
        title='Cost Comparison: RAG Strategies vs Baseline',
        xaxis_title='Strategy',
        yaxis_title='Cost (USD)',
        barmode='stack',
        height=500,
        showlegend=True,
        template='plotly_white'
    )
    
    return fig


def create_token_comparison_chart(results):
    """Create token usage comparison"""
    
    strategies = []
    input_tokens = []
    output_tokens = []
    
    for key, data in results.items():
        usage = data['usage']
        strategies.append(data['strategy'])
        input_tokens.append(usage.get('input_tokens', 0))
        output_tokens.append(usage.get('output_tokens', 0))
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Input Tokens',
        x=strategies,
        y=input_tokens,
        marker_color='steelblue',
        text=[f'{t:,}' for t in input_tokens],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='Output Tokens',
        x=strategies,
        y=output_tokens,
        marker_color='coral',
        text=[f'{t:,}' for t in output_tokens],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Token Usage Comparison',
        xaxis_title='Strategy',
        yaxis_title='Tokens',
        barmode='group',
        height=500,
        template='plotly_white'
    )
    
    return fig


def create_savings_chart(results):
    """Create cost savings visualization"""
    
    baseline_cost = results['baseline_full_doc']['usage'].get('total_cost', 0)
    
    strategies = []
    savings_pct = []
    colors = []
    
    for key, data in results.items():
        if key == 'baseline_full_doc':
            continue
        
        cost = data['usage'].get('total_cost', 0)
        savings = ((baseline_cost - cost) / baseline_cost * 100) if baseline_cost > 0 else 0
        
        strategies.append(data['strategy'])
        savings_pct.append(savings)
        
        # Color based on savings
        if savings > 75:
            colors.append('green')
        elif savings > 50:
            colors.append('orange')
        else:
            colors.append('red')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=strategies,
        y=savings_pct,
        marker_color=colors,
        text=[f'{s:.1f}%' for s in savings_pct],
        textposition='outside',
        name='Cost Savings'
    ))
    
    fig.update_layout(
        title='Cost Savings vs Full Document Baseline',
        xaxis_title='RAG Strategy',
        yaxis_title='Cost Savings (%)',
        height=500,
        template='plotly_white',
        yaxis=dict(range=[0, 100])
    )
    
    # Add target line at 75%
    fig.add_hline(y=75, line_dash="dash", line_color="gray", 
                  annotation_text="Target: 75% savings")
    
    return fig


def create_chunks_comparison(results):
    """Compare number of chunks retrieved"""
    
    strategies = []
    chunks = []
    
    for key, data in results.items():
        if key == 'baseline_full_doc':
            continue
        
        strategies.append(data['strategy'])
        chunks.append(data.get('num_chunks_retrieved', 0))
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=strategies,
        y=chunks,
        marker_color='mediumpurple',
        text=chunks,
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Number of Chunks Retrieved per Strategy',
        xaxis_title='Strategy',
        yaxis_title='Chunks Retrieved',
        height=400,
        template='plotly_white'
    )
    
    return fig


def create_dashboard(results):
    """Create comprehensive dashboard"""
    
    # Create 2x2 subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Cost Comparison',
            'Token Usage',
            'Cost Savings (%)',
            'Chunks Retrieved'
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # Extract data
    strategies = []
    total_costs = []
    input_tokens = []
    output_tokens = []
    chunks_retrieved = []
    savings_pct = []
    
    baseline_cost = results['baseline_full_doc']['usage'].get('total_cost', 0)
    
    for key, data in results.items():
        usage = data['usage']
        strategies.append(data['strategy'][:20] + '...' if len(data['strategy']) > 20 else data['strategy'])
        total_costs.append(usage.get('total_cost', 0))
        input_tokens.append(usage.get('input_tokens', 0))
        output_tokens.append(usage.get('output_tokens', 0))
        
        if key != 'baseline_full_doc':
            cost = usage.get('total_cost', 0)
            savings = ((baseline_cost - cost) / baseline_cost * 100) if baseline_cost > 0 else 0
            chunks_retrieved.append(data.get('num_chunks_retrieved', 0))
            savings_pct.append(savings)
    
    # Row 1, Col 1: Cost comparison
    fig.add_trace(
        go.Bar(x=strategies, y=total_costs, name='Total Cost', 
               marker_color='lightcoral', showlegend=False),
        row=1, col=1
    )
    
    # Row 1, Col 2: Token usage
    fig.add_trace(
        go.Bar(x=strategies, y=input_tokens, name='Input Tokens',
               marker_color='steelblue', showlegend=False),
        row=1, col=2
    )
    
    # Row 2, Col 1: Savings
    fig.add_trace(
        go.Bar(x=strategies[:-1], y=savings_pct, name='Savings %',
               marker_color='green', showlegend=False),
        row=2, col=1
    )
    
    # Row 2, Col 2: Chunks
    fig.add_trace(
        go.Bar(x=strategies[:-1], y=chunks_retrieved, name='Chunks',
               marker_color='mediumpurple', showlegend=False),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="RAG Strategy Comparison Dashboard",
        height=800,
        showlegend=False,
        template='plotly_white'
    )
    
    # Update axes labels
    fig.update_yaxes(title_text="Cost ($)", row=1, col=1)
    fig.update_yaxes(title_text="Tokens", row=1, col=2)
    fig.update_yaxes(title_text="Savings (%)", row=2, col=1)
    fig.update_yaxes(title_text="Chunks", row=2, col=2)
    
    return fig


def main():
    """Generate all visualizations"""
    
    console.print("\n" + "="*70)
    console.print("📊 GENERATING VISUALIZATIONS")
    console.print("="*70)
    
    # Load results
    results = load_results()
    if not results:
        return
    
    console.print(f"\n✅ Loaded results for {len(results)} strategies")
    
    # Create output directory
    viz_dir = Path("results/visualizations")
    viz_dir.mkdir(exist_ok=True)
    
    # Generate charts
    console.print("\n📈 Creating charts...")
    
    # 1. Cost comparison
    console.print("   1/5 Cost comparison...")
    fig_cost = create_cost_comparison_chart(results)
    fig_cost.write_html(viz_dir / "cost_comparison.html")
    
    # 2. Token usage
    console.print("   2/5 Token usage...")
    fig_tokens = create_token_comparison_chart(results)
    fig_tokens.write_html(viz_dir / "token_usage.html")
    
    # 3. Savings
    console.print("   3/5 Cost savings...")
    fig_savings = create_savings_chart(results)
    fig_savings.write_html(viz_dir / "cost_savings.html")
    
    # 4. Chunks
    console.print("   4/5 Chunks retrieved...")
    fig_chunks = create_chunks_comparison(results)
    fig_chunks.write_html(viz_dir / "chunks_comparison.html")
    
    # 5. Dashboard
    console.print("   5/5 Comprehensive dashboard...")
    fig_dashboard = create_dashboard(results)
    fig_dashboard.write_html(viz_dir / "dashboard.html")
    
    console.print(f"\n✅ All visualizations saved to: {viz_dir}/")
    
    # List files
    console.print(f"\n📁 Generated files:")
    for html_file in sorted(viz_dir.glob("*.html")):
        console.print(f"   - {html_file.name}")
    
    console.print(f"\n💡 Open dashboard.html in your browser to see all results!")
    console.print(f"   File path: {viz_dir / 'dashboard.html'}")


if __name__ == "__main__":
    main()
