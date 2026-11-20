import os
import sys
from pathlib import Path

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
    "Evaluate Adani Enterprises exposure across energy, airports, and infrastructure following recent events",
    "Whats better sector right now - space or EV? Best company from each and full analysis",
    "Compare pharma sector vs IT sector - which has better outlook and top pick from each",
    "Best renewable energy stocks in India right now with analysis",
]

def print_banner():
    banner = """[bold cyan]
╔════════════════════════════════════════════════════════════════════════════════╗
║    🏛️  FINRAG: MULTI-AGENT FINANCIAL & GEOPOLITICAL INTELLIGENCE              ║
║         Indian Equities (NSE & BSE) with Gemini AI                            ║
╚════════════════════════════════════════════════════════════════════════════════╝
[/bold cyan]
[green]✔ Agents:[/green] Stock Agent | Sector Research Agent | News Agent | Quant Agent
[green]✔ News:[/green] Live Google News India / Yahoo Finance / Reuters
[green]✔ Quant:[/green] Live NSE/BSE Ratios (P/E, Market Cap, ROE, Debt/Equity, Revenue Growth)
[green]✔ Sectors:[/green] Defence, Banking, IT, Pharma, Auto/EV, Energy, Power, Infra, Metals, FMCG
[green]✔ Filings:[/green] Annual Reports (HAL, BDL, L&T, Reliance Power)
    """
    console.print(banner)

def display_results(result):
    if result.get("fundamentals"):
        table = Table(title="📊 Live NSE/BSE Market Fundamentals", header_style="bold magenta")
        table.add_column("Ticker", style="cyan")
        table.add_column("Company", style="white")
        table.add_column("Price", justify="right", style="green")
        table.add_column("P/E", justify="right", style="yellow")
        table.add_column("Market Cap", justify="right", style="blue")

        for item in result["fundamentals"]:
            price_str = f"₹{item['price']:,.2f}" if isinstance(item['price'], (int, float)) else str(item['price'])
            pe_str = f"{item['pe']:.2f}" if isinstance(item['pe'], (int, float)) else str(item['pe'])
            table.add_row(item["ticker"], item["name"], price_str, pe_str, str(item["mcap"]))
        console.print(table)
        console.print("\n")

    console.print(Panel(
        Markdown(result.get("answer", "")),
        title="🧠 Multi-Agent Research Dossier",
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
        console.print("  [cyan]C.[/cyan] Custom Question (type any Indian stock / sector question)")
        console.print("  [cyan]Q.[/cyan] Quit")

        choice = input(f"\nEnter choice (1-{len(PRESET_QUERIES)}, C, or Q): ").strip()
        if choice.upper() == 'Q':
            console.print("[bold red]Exiting FinRAG. Happy investing![/bold red]")
            break

        if choice.isdigit() and 1 <= int(choice) <= len(PRESET_QUERIES):
            query = PRESET_QUERIES[int(choice) - 1]
        elif choice.upper() == 'C':
            query = input("Enter your custom query: ").strip()
            if not query:
                continue
        else:
            console.print("[red]Invalid selection. Try again.[/red]")
            continue

        console.print(f"\n[bold green]Running Multi-Agent Analysis on:[/bold green] '{query}'...")
        with console.status("[bold cyan]Gathering live news, fetching fundamentals, running sector research...[/bold cyan]"):
            result = orchestrator.run_comparative_analysis(query)

        display_results(result)

if __name__ == "__main__":
    main()
