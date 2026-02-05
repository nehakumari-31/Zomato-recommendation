import os
import sys
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel

from typing import Dict, List, Optional
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

load_dotenv()

from phase1.data_loader import ZomatoDataLoader
from phase2.input_validation import (
    validate_user_input,
    ValidationError,
    PricePreference,
    ValidatedUserInput,
)
from phase3.groq_client import GroqClient, GroqConfig, GroqError
from phase4.recommendation_engine import RecommendationEngine, RecommendedRestaurant

# Global console instance for CLI execution. By default, it prints to stdout/stderr.
console = Console()

def get_user_input(console: Console) -> Dict[str, str]:
    """
    Prompts user for city, cuisine, and price.
    """
    console.print(Panel("[bold blue]Enter your preferences for restaurant recommendations:[/bold blue]"))
    city = console.input("[bold green]City (e.g., Bangalore):[/bold green] ").strip()
    cuisine = console.input("[bold green]Cuisine (e.g., North Indian):[/bold green] ").strip()
    price = console.input("[bold green]Price for two (e.g., 500, 500-1000, budget):[/bold green] ").strip()
    return {"city": city, "cuisine": cuisine, "price": price}


def display_recommendations(recommendations: List['RecommendedRestaurant'], console: Console):
    """
    Displays a list of recommended restaurants in a formatted table.
    """
    if not recommendations:
        console.print(Panel("[bold red]No recommendations found for your criteria.[/bold red] Please try different inputs."))
        return

    table = Table(title="[bold magenta]Top Restaurant Recommendations[/bold magenta]")
    table.add_column("Name", style="bold cyan", justify="left")
    table.add_column("Rating", style="yellow", justify="center")
    table.add_column("Cost (for 2)", style="green", justify="right")
    table.add_column("Cuisines", style="white", justify="left")
    table.add_column("Address", style="dim white", justify="left")
    table.add_column("Reason", style="blue", justify="left")

    for rec in recommendations:
        rating_str = f"{rec.rating:.1f}/5" if rec.rating is not None else "N/A"
        cost_str = f"₹{rec.cost_for_two:.0f}" if rec.cost_for_two is not None else "N/A"
        cuisines_str = ", ".join(rec.cuisines) if rec.cuisines else "N/A"
        reason_str = rec.reason if rec.reason else "-"

        table.add_row(
            rec.name,
            rating_str,
            cost_str,
            cuisines_str,
            rec.address,
            reason_str
        )
    console.print(table)
    console.print("[dim]Powered by Zomato Data and Groq LLM[/dim]")


def run_cli_recommendation():
    """
    Main function to run the CLI recommendation service.
    """
    console.print(Panel("[bold white on blue]Zomato AI Restaurant Recommendation Service[/bold white on blue]"))

    # Phase 1: Data Loading
    with console.status("[bold green]Loading and processing Zomato data (Phase 1)...[/bold green]"):
        try:
            data_loader = ZomatoDataLoader()
            data_loader.load_dataset(split="train")
            processed_df = data_loader.clean_and_validate()
            available_cities = data_loader.get_unique_cities()
            available_cuisines = data_loader.get_unique_cuisines()
            console.log("[green]Phase 1: Data loaded and processed.[/green]")
        except Exception as e:
            console.print(f"[bold red]Error during Phase 1: {e}[/bold red]")
            return

    # Phase 2: User Input and Validation
    user_inputs = get_user_input(console)
    try:
        validated_input = validate_user_input(
            city=user_inputs["city"],
            cuisine=user_inputs["cuisine"],
            price=user_inputs["price"],
            available_cities=available_cities,
            available_cuisines=available_cuisines,
        )
        console.log("[green]Phase 2: User input validated.[/green]")
    except ValidationError as e:
        console.print(f"[bold red]Input Error: {e}[/bold red]")
        return

    # Phase 3 & 4: Recommendation Engine (includes Groq LLM call)
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        console.print(Panel(
            "[bold red]Error: GROQ_API_KEY environment variable is not set.[/bold red]\n"
            "Please set it in your shell (export GROQ_API_KEY=\'YOUR_KEY\') or in the .env file."
        ))
        return

    groq_client = None
    try:
        with console.status("[bold blue]Initializing Groq client (Phase 3)...[/bold blue]"):
            groq_client = GroqClient(GroqConfig(api_key=groq_api_key))
        console.log("[blue]Phase 3: Groq client initialized.[/blue]")

        with console.status("[bold purple]Generating recommendations (Phase 4)...[/bold purple]"):
            engine = RecommendationEngine(data_loader, groq_client, console=console) # Pass console here
            recommendations = engine.get_recommendations(validated_input)
        console.log("[purple]Phase 4: Recommendations generated.[/purple]")

    except GroqError as e:
        console.print(f"[bold red]Groq LLM Error: {e}[/bold red]")
        # Fallback to deterministic ranking if LLM fails here too
        console.print("[yellow]Attempting to get deterministic recommendations due to LLM error...[/yellow]")
        engine = RecommendationEngine(data_loader, GroqClient(GroqConfig(api_key="dummy")), console=console) # Pass console here
        recommendations = engine.get_recommendations(validated_input)
        if not recommendations:
            console.print("[bold red]Deterministic fallback also failed to find recommendations.[/bold red]")
            return

    except Exception as e:
        console.print(f"[bold red]Error during Phase 3/4: {e}[/bold red]")
        return
    finally:
        if groq_client:
            groq_client.close()

    # Phase 5: Display
    console.print("\n" + "="*70)
    display_recommendations(recommendations, console=console)
    console.print("="*70 + "\n")


if __name__ == "__main__":
    run_cli_recommendation()
