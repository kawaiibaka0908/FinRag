"""
Vortex-India: Interactive Multi-Agent Financial & Geopolitical Intelligence Terminal.
Analyzes NSE & BSE equities by combining real-time news, live fundamentals, and annual report filings.
"""
import os
import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))

from finrag.orchestrator.india_orchestrator import IndiaMultiAgentOrchestrator
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

console = Console()

PRESET_QUERIES = [
    "Compare BDL vs HAL on global affairs, defense orders, and recent geopolitical outlook",
    "Compare ICICI Bank with HDFC Bank and SBI on Net Interest Margins, valuation, and asset quality",
    "How do recent macro events affect Reliance and Bharti Airtel in the telecom and digital landscape?",
    "Analyze Larsen & Toubro (LT) order book growth and infrastructure execution in India",
    "Evaluate Adani Enterprises exposure across energy, airports, and infrastructure following recent events"
]

def print_banner():
    banner_text = """[bold cyan]
╔════════════════════════════════════════════════════════════════════════════════╗
║    🏛️  VORTEX-INDIA: MULTI-AGENT FINANCIAL & GEOPOLITICAL INTELLIGENCE        ║
║         Specialized for Indian Equities (NSE & BSE) with Gemini 2.0/3.8        ║
╚════════════════════════════════════════════════════════════════════════════════╝
[/bold cyan]
[green]✔ Multi-Agent Pipeline:[/green] Defence | Banking | Conglomerates | Infrastructure
[green]✔ Real-Time Ingestion:[/green] Reuters / Business Standard / Google News India
[green]✔ Quantitative Engine:[/green] Live NSE/BSE Ratios (P/E, Market Cap, ROE, Debt/Equity)
[green]✔ Corporate Filings:[/green] Integrated Annual Reports (HAL, BDL, L&T, Reliance)
    """
    console.print(banner_text)

def display_results(result: dict):
    # 1. Print Quant Summary Table
    if result.get("fundamentals"):
        table = Table(title="📊 Real-Time NSE/BSE Market Fundamentals", header_style="bold magenta")
        table.add_column("Ticker", style="cyan")
        table.add_column("Company Name", style="white")
        table.add_column("Live Price", justify="right", style="green")
        table.add_column("P/E Ratio", justify="right", style="yellow")
        table.add_column("Market Cap", justify="right", style="blue")

        for item in result["fundamentals"]:
            price_str = f"₹{item['price']:,.2f}" if isinstance(item['price'], (int, float)) else str(item['price'])
            pe_str = f"{item['pe']:.2f}" if isinstance(item['pe'], (int, float)) else str(item['pe'])
            table.add_row(item["ticker"], item["name"], price_str, pe_str, str(item["mcap"]))
        console.print(table)
        console.print("\n")

    # 2. Print Synthesis Markdown
    console.print(Panel(
        Markdown(result.get("answer", "")),
        title=f"🧠 Multi-Agent Research Dossier",
        border_style="bold blue",
        expand=False
    ))

def main():
    print_banner()
    orchestrator = IndiaMultiAgentOrchestrator()

    while True:
        console.print("\n[bold yellow]Select an option:[/bold yellow]")
        for idx, q in enumerate(PRESET_QUERIES, 1):
            console.print(f"  [cyan]{idx}.[/cyan] {q}")
        console.print("  [cyan]C.[/cyan] Custom Question (type any Indian stock question)")
        console.print("  [cyan]Q.[/cyan] Quit")

        choice = input("\nEnter choice (1-5, C, or Q): ").strip()
        if choice.upper() == 'Q':
            console.print("[bold red]Exiting Vortex-India. Happy investing![/bold red]")
            break

        if choice in ['1', '2', '3', '4', '5']:
            query = PRESET_QUERIES[int(choice) - 1]
        elif choice.upper() == 'C':
            query = input("Enter your custom query: ").strip()
            if not query:
                continue
        else:
            console.print("[red]Invalid selection. Try again.[/red]")
            continue

        console.print(f"\n[bold green]Running Multi-Agent Analysis on:[/bold green] '{query}'...")
        with console.status("[bold cyan]Gathering live news, fetching fundamentals, and synthesizing research...[/bold cyan]"):
            result = orchestrator.run_comparative_analysis(query)

        display_results(result)

if __name__ == "__main__":
    main()
