from __future__ import annotations

import threading
import webbrowser
from pathlib import Path
from typing import Literal

import typer
import uvicorn

from .dataset import Dataset, DatasetError
from .web import create_app

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    dataset_yaml: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    category: Literal["detection", "segmentation"] = typer.Option(..., "--category", case_sensitive=False),
    host: str = typer.Option("127.0.0.1", help="Address to bind"),
    port: int = typer.Option(8765, min=1, max=65535),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not open a browser automatically"),
) -> None:
    """Start the workbench for DATASET_YAML."""
    try:
        dataset = Dataset(dataset_yaml, category)
    except DatasetError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(2) from exc
    url = f"http://{host}:{port}"
    typer.echo(f"Indexed {len(dataset.images)} images. Workbench: {url}")
    if not no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(dataset), host=host, port=port)


if __name__ == "__main__":
    app()
