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
    dataset_yaml: Path = typer.Argument(..., exists=True, dir_okay=True, readable=True, help="Dataset YAML, or the dataset root directory for classification"),
    category: Literal["detection", "segmentation", "classification"] = typer.Option(..., "--category", case_sensitive=False),
    host: str = typer.Option("127.0.0.1", help="Address to bind"),
    port: int = typer.Option(8765, min=1, max=65535),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not open a browser automatically"),
    model: Path | None = typer.Option(None, "--model", help="Ultralytics model (.pt/.onnx) to preload for assisted labeling"),
) -> None:
    """Start the workbench for DATASET_YAML."""
    if dataset_yaml.is_dir() and category != "classification":
        typer.echo("Error: a directory dataset requires --category classification", err=True)
        raise typer.Exit(2)
    try:
        dataset = Dataset(dataset_yaml, category, background_probe=True)
    except DatasetError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(2) from exc
    url = f"http://{host}:{port}"
    typer.echo(f"Indexed {len(dataset.images)} images. Workbench: {url}")
    app_instance = create_app(dataset)
    if model is not None:
        try:
            app_instance.state.predictor.load(str(model))
            typer.echo(f"Loaded model {model}")
        except DatasetError as exc:
            typer.echo(f"Warning: {exc}", err=True)
    if not no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app_instance, host=host, port=port)


if __name__ == "__main__":
    app()
