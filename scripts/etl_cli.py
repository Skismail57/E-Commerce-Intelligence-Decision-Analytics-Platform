from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from config.logging_config import get_logger
from config.settings import settings

console = Console()
app = typer.Typer(
    help="E-Commerce Intelligence & Decision Analytics Platform CLI",
    add_completion=False,
    rich_markup_mode="rich",
)
logger = get_logger(__name__)


def _do_generate(
    num_customers: int = None,
    num_products: int = None,
    num_orders: int = None,
    seed: int = None,
    start_date: str = None,
    end_date: str = None,
    save: bool = True,
):
    from src.ingestion.generate_synthetic_data import SyntheticDataGenerator

    console.rule("[bold purple]SYNTHETIC DATA GENERATOR[/bold purple]")

    generator = SyntheticDataGenerator(
        random_state=seed,
        start_date=start_date,
        end_date=end_date,
        num_customers=num_customers,
        num_products=num_products,
        num_orders=num_orders,
    )
    datasets = generator.generate_all()

    table = Table(title="Generated Datasets")
    table.add_column("Table", style="cyan", no_wrap=True)
    table.add_column("Rows", justify="right", style="green")
    table.add_column("Memory", justify="right", style="yellow")
    for name, df in datasets.items():
        mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        table.add_row(name, f"{len(df):,}", f"{mem_mb:.1f} MB")
    console.print(table)

    if save:
        generator.save_to_csv()
        console.print(f"[green]Datasets saved to[/green] data/raw/")

    return datasets


def _do_validate(data_dir: Path = None, tables: list = None):
    from src.validation.schemas import DataValidator

    console.rule("[bold blue]DATA VALIDATION[/bold blue]")
    validator = DataValidator()
    directory = data_dir or settings.RAW_DATA_DIR
    results = validator.validate_all(directory, tables=tables)
    t = Table(title="Validation Results")
    t.add_column("Table", style="cyan")
    t.add_column("Rows", justify="right", style="green")
    t.add_column("Status", style="bold")
    for name, df in results.items():
        t.add_row(name, f"{len(df):,}", "[green]PASS[/green]")
    console.print(t)
    return results


def _do_profile(data_dir: Path = None, tables: list = None, output_format: str = None):
    from src.cleaning.data_profiler import DataProfiler

    console.rule("[bold magenta]DATA PROFILING[/bold magenta]")
    directory = data_dir or settings.RAW_DATA_DIR
    profiler = DataProfiler()
    report = profiler.profile_all(directory, tables=tables)

    t = Table(title="Data Profile Summary")
    t.add_column("Table", style="cyan")
    t.add_column("Rows", justify="right")
    t.add_column("Cols", justify="right")
    t.add_column("Nulls", justify="right")
    t.add_column("Null %", justify="right", style="yellow")
    t.add_column("Dup %", justify="right", style="red")
    for name, stats in report.items():
        t.add_row(
            name,
            f"{stats['rows']:,}",
            str(stats["cols"]),
            f"{stats['null_count']:,}",
            f"{stats['null_pct']:.2f}%",
            f"{stats['duplicate_pct']:.2f}%",
        )
    console.print(t)

    if output_format:
        from src.cleaning.data_profiler import export_profile_report
        out_path = export_profile_report(profiler, output_format, directory.parent)
        console.print(f"[green]Detailed report saved to[/green] {out_path}")

    return report


def _do_db_check():
    from src.ingestion.database import DatabaseManager

    console.rule("[bold green]DATABASE CONNECTION TEST[/bold green]")
    db = DatabaseManager()
    try:
        db.test_connection()
        console.print("[green]Database connection: OK[/green]")
    except Exception as e:
        console.print(f"[red]Database connection: FAILED[/red]\n{e}")
        raise typer.Exit(code=1)


def _do_load(data_dir: Path = None, reset: bool = False, tables: list = None, if_exists: str = "append"):
    from src.ingestion.data_loader import DataLoader

    console.rule("[bold orange]DATA LOAD TO POSTGRESQL[/bold orange]")
    loader = DataLoader(data_dir=data_dir)
    loader.db.test_connection()

    if reset:
        loader.reset_database()

    results = loader.load_all(tables=tables, if_exists=if_exists)

    t = Table(title="Load Summary")
    t.add_column("Table", style="cyan")
    t.add_column("Rows Loaded", justify="right", style="green")
    for n, c in results.items():
        t.add_row(n, f"{c:,}")
    console.print(t)
    return results


def _do_fk_integrity(data_dir: Path = None):
    from src.validation.integrity import FKIntegrityChecker

    console.rule("[bold yellow]FOREIGN KEY INTEGRITY CHECK[/bold yellow]")
    directory = data_dir or settings.RAW_DATA_DIR
    checker = FKIntegrityChecker()
    report = checker.check_all(directory)
    ok = checker.print_report(report, console)
    if not ok:
        raise typer.Exit(code=1)
    return report


def _do_clean(raw_dir: Path = None, staging_dir: Path = None, tables: list = None):
    from src.cleaning.cleaner import DataCleaner

    console.rule("[bold cyan]DATA CLEANING PIPELINE[/bold cyan]")
    raw = raw_dir or settings.RAW_DATA_DIR
    staging = staging_dir or settings.STAGING_DATA_DIR
    cleaner = DataCleaner()
    results = cleaner.clean_all(raw, staging, tables=tables)
    t = Table(title="Cleaning Summary")
    t.add_column("Table", style="cyan")
    t.add_column("In Rows", justify="right", style="yellow")
    t.add_column("Out Rows", justify="right", style="green")
    t.add_column("Removed", justify="right", style="red")
    t.add_column("Issues Fixed", justify="right")
    for n, r in results.items():
        t.add_row(
            n,
            f"{r['in_rows']:,}",
            f"{r['out_rows']:,}",
            f"{r['removed']:,}",
            f"{r['issues_fixed']:,}",
        )
    console.print(t)
    return results


def _do_transform(staging_dir: Path = None, processed_dir: Path = None):
    from src.transformation.feature_engineering import FeatureEngineer
    from src.transformation.aggregator import DataAggregator

    console.rule("[bold violet]FEATURE ENGINEERING & TRANSFORMATION[/bold violet]")
    staging = staging_dir or settings.STAGING_DATA_DIR
    processed = processed_dir or settings.PROCESSED_DATA_DIR

    fe = FeatureEngineer()
    transform_results = fe.run_all_transformations(staging, processed, save=True)
    t1 = Table(title="Feature Engineering Results")
    t1.add_column("Dataset", style="cyan")
    t1.add_column("Rows", justify="right", style="green")
    t1.add_column("Cols", justify="right", style="yellow")
    for name, df in transform_results.items():
        t1.add_row(name, f"{len(df):,}", str(len(df.columns)))
    console.print(t1)

    console.print()
    console.rule("[bold violet]DATA AGGREGATION[/bold violet]")
    agg = DataAggregator()
    agg_results = agg.run_all_aggregations(staging, processed, save=True)
    t2 = Table(title="Aggregation Results")
    t2.add_column("Dataset", style="cyan")
    t2.add_column("Rows", justify="right", style="green")
    t2.add_column("Cols", justify="right", style="yellow")
    for name, df in agg_results.items():
        t2.add_row(name, f"{len(df):,}", str(len(df.columns)))
    console.print(t2)

    return {**transform_results, **agg_results}


def _do_customer_analytics(staging_dir: Path = None, processed_dir: Path = None):
    from src.analytics.customer_analytics import CustomerIntelligence

    console.rule("[bold blue]CUSTOMER INTELLIGENCE ANALYTICS[/bold blue]")
    staging = staging_dir or settings.STAGING_DATA_DIR
    ci = CustomerIntelligence(staging)
    results = ci.run_all_customer_analytics(staging, processed_dir)

    t = Table(title="Customer Intelligence Summary")
    t.add_column("Artifact", style="cyan")
    t.add_column("Details", style="green")
    for name, value in results.items():
        if isinstance(value, pd.DataFrame):
            t.add_row(name, f"{len(value):,} rows x {len(value.columns)} cols")
        elif isinstance(value, dict):
            t.add_row(name, "dict summary")
        else:
            t.add_row(name, str(value))
    console.print(t)

    if "rfm" in results:
        rfm = results["rfm"]
        seg_counts = rfm["rfm_segment"].value_counts()
        t2 = Table(title="RFM Segment Distribution")
        t2.add_column("Segment", style="magenta")
        t2.add_column("Count", justify="right", style="green")
        t2.add_column("%", justify="right", style="yellow")
        for seg, cnt in seg_counts.items():
            pct = round(cnt / len(rfm) * 100, 2)
            t2.add_row(seg, f"{cnt:,}", f"{pct}%")
        console.print(t2)

    if "pareto" in results:
        pareto = results["pareto"]
        t3 = Table(title="Pareto Analysis")
        t3.add_column("Metric", style="cyan")
        t3.add_column("Value", style="bold green")
        t3.add_row("Top 10% Profit Share", f"{pareto['top_10pct_profit_share_pct']}%")
        t3.add_row("Top 20% Profit Share", f"{pareto['top_20pct_profit_share_pct']}%")
        t3.add_row("Top 50% Profit Share", f"{pareto['top_50pct_profit_share_pct']}%")
        t3.add_row("Total Customers", f"{pareto['total_customers']:,}")
        console.print(t3)

    return results


def _do_product_analytics(staging_dir: Path = None, processed_dir: Path = None):
    from src.analytics.product_analytics import ProductIntelligence

    console.rule("[bold purple]PRODUCT INTELLIGENCE ANALYTICS[/bold purple]")
    staging = staging_dir or settings.STAGING_DATA_DIR
    pi = ProductIntelligence(staging)
    results = pi.run_all_product_analytics(staging, processed_dir)

    t = Table(title="Product Intelligence Summary")
    t.add_column("Artifact", style="cyan")
    t.add_column("Details", style="green")
    for name, value in results.items():
        if isinstance(value, pd.DataFrame):
            t.add_row(name, f"{len(value):,} rows x {len(value.columns)} cols")
        else:
            t.add_row(name, str(value))
    console.print(t)

    if "product_matrix" in results:
        pm = results["product_matrix"]
        quad_counts = pm["quadrant"].value_counts()
        t2 = Table(title="Product Matrix Quadrants")
        t2.add_column("Quadrant", style="magenta")
        t2.add_column("Products", justify="right", style="green")
        t2.add_column("%", justify="right", style="yellow")
        for q, cnt in quad_counts.items():
            pct = round(cnt / len(pm) * 100, 2)
            t2.add_row(q, f"{cnt:,}", f"{pct}%")
        console.print(t2)

    if "lifecycle" in results:
        lc = results["lifecycle"]
        stage_counts = lc["lifecycle_stage"].value_counts()
        t3 = Table(title="Product Lifecycle Stages")
        t3.add_column("Stage", style="magenta")
        t3.add_column("Count", justify="right", style="green")
        for s, cnt in stage_counts.items():
            t3.add_row(s, f"{cnt:,}")
        console.print(t3)

    return results


def _do_marketing_analytics(staging_dir: Path = None, processed_dir: Path = None):
    from src.analytics.marketing_analytics import MarketingAnalyzer

    console.rule("[bold green]MARKETING ANALYTICS[/bold green]")
    staging = staging_dir or settings.STAGING_DATA_DIR
    ma = MarketingAnalyzer(staging)
    results = ma.run_all_marketing_analytics(staging, processed_dir)

    t = Table(title="Marketing Analytics Summary")
    t.add_column("Artifact", style="cyan")
    t.add_column("Details", style="green")
    for name, value in results.items():
        if isinstance(value, pd.DataFrame):
            t.add_row(name, f"{len(value):,} rows x {len(value.columns)} cols")
        elif isinstance(value, dict):
            t.add_row(name, "overall funnel metrics")
        else:
            t.add_row(name, str(value))
    console.print(t)

    if "overall_funnel" in results:
        f = results["overall_funnel"]
        t2 = Table(title="Overall Conversion Funnel")
        t2.add_column("Step", style="magenta")
        t2.add_column("Value", justify="right", style="bold green")
        t2.add_row("Sessions", f"{f['total_sessions']:,}")
        t2.add_row("Cart Adds", f"{f['total_cart_adds']:,}")
        t2.add_row("Checkouts Started", f"{f['total_checkouts_started']:,}")
        t2.add_row("Checkouts Completed", f"{f['total_checkouts_completed']:,}")
        t2.add_row("Orders", f"{f['total_orders']:,}")
        t2.add_row("Session → Cart", f"{f['session_to_cart_pct']}%")
        t2.add_row("Cart → Checkout", f"{f['cart_to_checkout_pct']}%")
        t2.add_row("Checkout → Order", f"{f['checkout_to_order_pct']}%")
        t2.add_row("Session → Order", f"{f['session_to_order_pct']}%")
        console.print(t2)

    return results


def _do_analytics_all(staging_dir: Path = None, processed_dir: Path = None):
    results = {}
    results["customer"] = _do_customer_analytics(staging_dir, processed_dir)
    results["product"] = _do_product_analytics(staging_dir, processed_dir)
    results["marketing"] = _do_marketing_analytics(staging_dir, processed_dir)
    console.rule("[bold blue_violet]ALL ANALYTICS COMPLETE[/bold blue_violet]")
    return results


def _do_deploy_sql_views():
    from src.ingestion.database import DatabaseManager

    console.rule("[bold red]DEPLOYING SQL VIEWS TO POSTGRESQL[/bold red]")
    db = DatabaseManager()
    db.test_connection()

    sql_files = [
        settings.SQL_DIR / "staging" / "002_staging_views.sql",
        settings.SQL_DIR / "kpis" / "001_executive_kpis.sql",
        settings.SQL_DIR / "analytics" / "001_advanced_analytics.sql",
        settings.SQL_DIR / "transformations" / "001_advanced_transformation_views.sql",
    ]

    results = []
    for sql_file in sql_files:
        if sql_file.exists():
            try:
                db.read_sql_file(str(sql_file))
                results.append((sql_file.name, "OK"))
                logger.info(f"Deployed {sql_file}")
            except Exception as e:
                results.append((sql_file.name, f"FAIL: {e}"))
                logger.error(f"Failed to deploy {sql_file}: {e}")
        else:
            results.append((sql_file.name, "SKIP (not found)"))

    t = Table(title="SQL View Deployment")
    t.add_column("SQL File", style="cyan")
    t.add_column("Status", style="bold")
    for name, status in results:
        color = "green" if status == "OK" else "red" if "FAIL" in status else "yellow"
        t.add_row(name, f"[{color}]{status}[/{color}]")
    console.print(t)

    return results


@app.command()
def generate(
    num_customers: Optional[int] = typer.Option(None, "--customers", "-c", help="Number of customers"),
    num_products: Optional[int] = typer.Option(None, "--products", "-p", help="Number of products"),
    num_orders: Optional[int] = typer.Option(None, "--orders", "-o", help="Number of orders"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed"),
    start_date: Optional[str] = typer.Option(None, "--start", help="Start date YYYY-MM-DD"),
    end_date: Optional[str] = typer.Option(None, "--end", help="End date YYYY-MM-DD"),
    save: bool = typer.Option(True, "--save/--no-save", help="Save CSVs to data/raw"),
) -> None:
    """Generate synthetic e-commerce dataset."""
    _do_generate(num_customers, num_products, num_orders, seed, start_date, end_date, save)


@app.command()
def validate(
    data_dir: Optional[str] = typer.Option(None, "--dir", "-d", help="Directory with CSVs"),
    tables: Optional[str] = typer.Option(None, "--tables", "-t", help="Comma-separated table names"),
) -> None:
    """Validate datasets using Pandera schemas."""
    table_list = tables.split(",") if tables else None
    directory = Path(data_dir) if data_dir else None
    _do_validate(directory, table_list)


@app.command()
def profile(
    data_dir: Optional[str] = typer.Option(None, "--dir", "-d", help="Directory with CSVs"),
    tables: Optional[str] = typer.Option(None, "--tables", "-t", help="Comma-separated table names"),
    report: Optional[str] = typer.Option(None, "--report", "-r", help="Export detailed report: html/json/both"),
) -> None:
    """Run a data profile on generated CSVs (with detailed report option)."""
    table_list = tables.split(",") if tables else None
    directory = Path(data_dir) if data_dir else None
    _do_profile(directory, table_list, report)


@app.command()
def db_check() -> None:
    """Test PostgreSQL database connection."""
    _do_db_check()


@app.command()
def load(
    data_dir: Optional[str] = typer.Option(None, "--dir", "-d"),
    reset: bool = typer.Option(False, "--reset", help="Drop and recreate schema first"),
    tables: Optional[str] = typer.Option(None, "--tables", "-t"),
    if_exists: str = typer.Option("append", "--if-exists", "-m", help="append/replace/fail"),
) -> None:
    """Load generated CSVs into PostgreSQL warehouse."""
    table_list = tables.split(",") if tables else None
    directory = Path(data_dir) if data_dir else None
    _do_load(directory, reset, table_list, if_exists)


@app.command()
def check_integrity(
    data_dir: Optional[str] = typer.Option(None, "--dir", "-d", help="Directory with CSVs"),
) -> None:
    """Run cross-table foreign key integrity checks."""
    directory = Path(data_dir) if data_dir else None
    _do_fk_integrity(directory)


@app.command()
def clean(
    raw_dir: Optional[str] = typer.Option(None, "--raw", help="Raw data directory"),
    staging_dir: Optional[str] = typer.Option(None, "--staging", help="Output staging directory"),
    tables: Optional[str] = typer.Option(None, "--tables", "-t", help="Comma-separated table names"),
) -> None:
    """Run cleaning pipeline (dedup, types, outliers) raw -> staging."""
    table_list = tables.split(",") if tables else None
    r = Path(raw_dir) if raw_dir else None
    s = Path(staging_dir) if staging_dir else None
    _do_clean(r, s, table_list)


@app.command()
def transform(
    staging_dir: Optional[str] = typer.Option(None, "--staging", help="Staging data directory"),
    processed_dir: Optional[str] = typer.Option(None, "--processed", help="Output processed directory"),
) -> None:
    """Run feature engineering + data aggregations: staging -> processed."""
    s = Path(staging_dir) if staging_dir else None
    p = Path(processed_dir) if processed_dir else None
    _do_transform(s, p)


@app.command()
def customer_analytics(
    staging_dir: Optional[str] = typer.Option(None, "--staging", help="Staging data directory"),
    processed_dir: Optional[str] = typer.Option(None, "--processed", help="Output processed directory"),
) -> None:
    """Run Customer Intelligence: RFM, CLV, cohort, Pareto, churn features."""
    s = Path(staging_dir) if staging_dir else None
    p = Path(processed_dir) if processed_dir else None
    _do_customer_analytics(s, p)


@app.command()
def product_analytics(
    staging_dir: Optional[str] = typer.Option(None, "--staging", help="Staging data directory"),
    processed_dir: Optional[str] = typer.Option(None, "--processed", help="Output processed directory"),
) -> None:
    """Run Product Intelligence: 360 view, matrix, lifecycle, price elasticity."""
    s = Path(staging_dir) if staging_dir else None
    p = Path(processed_dir) if processed_dir else None
    _do_product_analytics(s, p)


@app.command()
def marketing_analytics(
    staging_dir: Optional[str] = typer.Option(None, "--staging", help="Staging data directory"),
    processed_dir: Optional[str] = typer.Option(None, "--processed", help="Output processed directory"),
) -> None:
    """Run Marketing Analytics: funnel, campaign performance, CAC, ROAS."""
    s = Path(staging_dir) if staging_dir else None
    p = Path(processed_dir) if processed_dir else None
    _do_marketing_analytics(s, p)


@app.command()
def analytics_all(
    staging_dir: Optional[str] = typer.Option(None, "--staging", help="Staging data directory"),
    processed_dir: Optional[str] = typer.Option(None, "--processed", help="Output processed directory"),
) -> None:
    """Run ALL analytics modules: customer + product + marketing."""
    s = Path(staging_dir) if staging_dir else None
    p = Path(processed_dir) if processed_dir else None
    _do_analytics_all(s, p)


@app.command()
def deploy_views() -> None:
    """Deploy all SQL views (staging, KPIs, analytics, transformations) to PostgreSQL."""
    _do_deploy_sql_views()


@app.command()
def run_pipeline(
    num_customers: int = typer.Option(5000, "--customers", "-c"),
    num_products: int = typer.Option(500, "--products", "-p"),
    num_orders: int = typer.Option(10000, "--orders", "-o"),
    seed: int = typer.Option(42, "--seed", "-s"),
    reset_db: bool = typer.Option(True, "--reset-db/--no-reset-db"),
    load_db: bool = typer.Option(True, "--load-db/--no-load-db"),
    report: str = typer.Option("html", "--report", help="Profile report format: none/html/json/both"),
    run_analytics: bool = typer.Option(True, "--analytics/--no-analytics", help="Run customer/product/marketing analytics"),
    run_transform: bool = typer.Option(True, "--transform/--no-transform", help="Run feature engineering & aggregation"),
    deploy_sql: bool = typer.Option(False, "--deploy-sql/--no-deploy-sql", help="Deploy SQL views to Postgres (requires running DB)"),
) -> None:
    """Run full end-to-end platform pipeline: GENERATE → CLEAN → VALIDATE → INTEGRITY → PROFILE → TRANSFORM → ANALYTICS → LOAD."""
    console.rule("[bold red]FULL E-COMMERCE INTELLIGENCE PIPELINE[/bold red]")

    _do_generate(num_customers, num_products, num_orders, seed, save=True)
    _do_clean()
    _do_validate(data_dir=settings.STAGING_DATA_DIR)
    _do_fk_integrity(data_dir=settings.STAGING_DATA_DIR)
    fmt = None if report == "none" else report
    _do_profile(data_dir=settings.STAGING_DATA_DIR, output_format=fmt)

    if run_transform:
        try:
            _do_transform()
        except Exception as e:
            console.print(f"[yellow]Skipping transform: {e}[/yellow]")

    if run_analytics:
        try:
            _do_analytics_all()
        except Exception as e:
            console.print(f"[yellow]Skipping analytics: {e}[/yellow]")

    if load_db:
        try:
            _do_db_check()
            _do_load(data_dir=settings.STAGING_DATA_DIR, reset=reset_db)
            if deploy_sql:
                _do_deploy_sql_views()
        except Exception as e:
            console.print(f"[yellow]Skipping DB load: {e}[/yellow]")

    console.rule("[bold green]PIPELINE COMPLETE[/bold green]")


if __name__ == "__main__":
    app()
