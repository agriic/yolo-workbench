from __future__ import annotations

import threading
import webbrowser
from pathlib import Path
from typing import Literal, Optional

import typer
import uvicorn

from .dataset import Dataset, DatasetError
from .web import create_app, create_workspace_app
from .workspace import Workspace

app = typer.Typer(add_completion=False, no_args_is_help=True)

LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]"}


@app.command()
def main(
    dataset_yaml: Optional[Path] = typer.Argument(None, exists=True, dir_okay=True, readable=True, help="Dataset YAML, or the dataset root directory for classification"),
    projects: Optional[Path] = typer.Option(None, "--projects", file_okay=False, help="Projects folder: create, edit, and delete datasets inside it"),
    category: Optional[Literal["detection", "segmentation", "classification"]] = typer.Option(None, "--category", case_sensitive=False, help="Dataset category (single-dataset mode)"),
    host: str = typer.Option("127.0.0.1", help="Address to bind"),
    port: int = typer.Option(8765, min=1, max=65535),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not open a browser automatically"),
    model: Optional[Path] = typer.Option(None, "--model", help="Ultralytics (.pt) or RF-DETR (.pth) model to preload for assisted labeling (single-dataset mode)"),
) -> None:
    """Start the workbench for DATASET_YAML, or for a folder of projects with --projects."""
    if (dataset_yaml is None) == (projects is None):
        typer.echo("Error: pass either a dataset (DATASET_YAML) or --projects DIR", err=True)
        raise typer.Exit(2)
    url = f"http://{host}:{port}"
    # binding a wildcard address means the user wants LAN access, so the Host check cannot apply
    allowed_hosts = None if host in {"0.0.0.0", "::"} else LOCAL_HOSTS | {host}
    guard = {"allowed_hosts": allowed_hosts, "require_header": True}
    if projects is not None:
        if category is not None or model is not None:
            typer.echo("Error: --category and --model apply to single-dataset mode; projects store their own category and models", err=True)
            raise typer.Exit(2)
        if not projects.exists() and not typer.confirm(f"Create projects folder {projects}?", default=True):
            raise typer.Exit(1)
        workspace = Workspace(projects)
        typer.echo(f"Projects folder {workspace.root}. Workbench: {url}")
        app_instance = create_workspace_app(workspace, **guard)
    else:
        if category is None:
            typer.echo("Error: --category is required with a dataset", err=True)
            raise typer.Exit(2)
        if dataset_yaml.is_dir() and category != "classification":
            typer.echo("Error: a directory dataset requires --category classification", err=True)
            raise typer.Exit(2)
        try:
            dataset = Dataset(dataset_yaml, category, background_probe=True)
        except DatasetError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(2) from exc
        typer.echo(f"Indexed {len(dataset.images)} images. Workbench: {url}")
        app_instance = create_app(dataset, **guard)
        if model is not None:
            try:
                app_instance.state.predictor.load(str(model))
                typer.echo(f"Loaded model {model}")
            except DatasetError as exc:
                typer.echo(f"Warning: {exc}", err=True)
    app_instance.state.training.start()  # advances the training queue; runs themselves are separate processes
    if not no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app_instance, host=host, port=port)


if __name__ == "__main__":
    app()
