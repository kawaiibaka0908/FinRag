import os
import sys
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))

from finrag.orchestrator.langgraph_orchestrator import FinRAGLangGraphOrchestrator
from finrag.vectorstore.annual_report_store import load_annual_report_store
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

SUGGESTED_THEMES = [
    "What is the better investment right now between the Space sector and the EV sector?",
    "Compare Hindustan Aeronautics (HAL) vs Bharat Dynamics (BDL) on order book and valuation multiples",
    "Compare ICICI Bank, HDFC Bank, and SBI on Net Interest Margins, credit costs, and asset quality",
    "Evaluate Reliance Industries and Adani Enterprises on clean energy capex and recent news",
    "Compare Indian IT sector vs Pharma sector for a 12-month defensive investment horizon"
]

def print_banner():
    banner = """[bold cyan]
╔════════════════════════════════════════════════════════════════════════════════╗
║    🏛️  FINRAG INSTITUTIONAL: LANGGRAPH EQUITY INTELLIGENCE TERMINAL            ║
║         Stateful Multi-Agent Graph (NSE/BSE) | RAG + Tool-Calling Workflows    ║
╚════════════════════════════════════════════════════════════════════════════════╝
[/bold cyan]
[green]● LangGraph Graph:[/green] Intent Node ──► Sector Node ──► Quant Engine ──► News Node ──► Filings RAG ──► Synthesis Node
[green]● Real-Time Feeds:[/green] Live NSE/BSE Exchange Quotes | Reuters / Google News India RSS
[green]● Corporate Filings:[/green] Integrated Annual Reports (HAL, BDL, L&T, Reliance Power)
    """
    console.print(banner)

def display_results(result):
    console.print(f"\n[bold green]⚡ LangGraph Workflow Completed in {result.get('elapsed_seconds', 'N/A')} seconds[/bold green]\n")

    # 1. LangGraph State Node Trace Table
    if result.get("trace"):
        trace_table = Table(title="🕸️ LangGraph Node-by-Node State Execution Audit Trail", header_style="bold cyan")
        trace_table.add_column("Node", style="dim", width=6)
        trace_table.add_column("Agent / Tool", style="bold yellow")
        trace_table.add_column("State Action", style="white")
        trace_table.add_column("Output Details", style="dim white")

        for t in result["trace"]:
            trace_table.add_row(str(t["step"]), t["agent"], t["action"], t["detail"])
        console.print(trace_table)
        console.print("\n")

    # 2. Quantitative Summary Table
    if result.get("fundamentals"):
        table = Table(title="📊 Real-Time Market Fundamentals & Valuation Ratios", header_style="bold magenta")
        table.add_column("Ticker", style="cyan")
        table.add_column("Company", style="white")
        table.add_column("Sector", style="dim")
        table.add_column("Price (₹)", justify="right", style="green")
        table.add_column("P/E", justify="right", style="yellow")
        table.add_column("Market Cap", justify="right", style="blue")
        table.add_column("ROE", justify="right", style="white")
        table.add_column("1Y Return", justify="right", style="magenta")

        for item in result["fundamentals"]:
            price_str = f"₹{item['price']:,.2f}" if isinstance(item['price'], (int, float)) else str(item['price'])
            pe_str = f"{item['pe']:.2f}" if isinstance(item['pe'], (int, float)) else str(item['pe'])
            table.add_row(
                item.get("ticker", "N/A"),
                item.get("name", "N/A")[:24],
                item.get("sector", "Equities")[:16],
                price_str,
                pe_str,
                str(item.get("mcap", "N/A")),
                str(item.get("roe", "N/A")),
                str(item.get("yr_return", "N/A"))
            )
        console.print(table)
        console.print("\n")

    # 3. Live News Feed Ingested
    if result.get("news"):
        console.print("[bold yellow]📰 Live Market News & Catalysts Ingested:[/bold yellow]")
        for idx, n in enumerate(result["news"][:4], 1):
            src = f" ({n.get('source', '')})" if n.get('source') else ""
            pub = f" [{n.get('published', '')}]" if n.get('published') else ""
            console.print(f"  [dim]{idx}.[/dim] {n.get('title')}[cyan]{src}[/cyan][dim]{pub}[/dim]")
        console.print("\n")

    # 4. Institutional Advisory Dossier
    console.print(Panel(
        Markdown(result.get("answer", "")),
        title="📑 Institutional Equity Research Dossier (LangGraph Output)",
        border_style="bold blue",
        expand=False
    ))
    if result.get("sources_used"):
        console.print(Panel(str(result["sources_used"]), title="Sources used for this query", border_style="dim"))

def main():
    print_banner()
    vector_store, store_message = load_annual_report_store()
    if store_message:
        console.print(f"[yellow]Annual report RAG unavailable:[/yellow] {store_message}")
    orchestrator = FinRAGLangGraphOrchestrator(vector_store=vector_store)

    while True:
        console.print("[bold white]Type your investment query, sector comparison, or 'examples' (or 'q' to quit):[/bold white]")
        user_input = input(">> ").strip()

        if not user_input:
            continue

        if user_input.lower() in ['q', 'exit', 'quit']:
            console.print("[bold red]Session terminated. Happy investing![/bold red]")
            break

        if user_input.lower() == 'examples':
            console.print("\n[bold yellow]Suggested Advisory Inquiries:[/bold yellow]")
            for idx, ex in enumerate(SUGGESTED_THEMES, 1):
                console.print(f"  [cyan]{idx}.[/cyan] {ex}")
            console.print("\n")
            continue

        console.print(f"\n[bold green]Executing LangGraph State Machine for:[/bold green] '{user_input}'...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("[bold cyan]Navigating graph nodes, extracting market ratios, synthesizing...", total=None)
            result = orchestrator.run(user_input)

        display_results(result)
        console.print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
