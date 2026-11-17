"""Execute the vignettes into notebooks, or check the notebooks are current.

The jupytext percent `.py` in `docs/vignettes/` is the source. `execute` runs
it in a fresh kernel and writes the paired `.ipynb` with its outputs, which
mkdocs-jupyter renders without executing. Run from `python/`:

    uv run --no-sync python dev/vignettes.py execute pbmc_single_cell
    uv run --no-sync python dev/vignettes.py check
"""

import time
from pathlib import Path

import click
import jupytext
import nbformat
from beartype import beartype
from nbclient import NotebookClient

###########
# Globals #
###########

VIGNETTE_DIR: Path = Path(__file__).resolve().parent.parent / "docs" / "vignettes"

# Generous: the notebooks download their data on the first run.
CELL_TIMEOUT_S: int = 3600


#############
# Functions #
#############


@beartype
def _sources(names: tuple[str, ...]) -> list[Path]:
    """The vignette sources by name, all of them when no name is given."""
    if not names:
        return sorted(VIGNETTE_DIR.glob("*.py"))
    paths = [VIGNETTE_DIR / f"{n.removesuffix('.py')}.py" for n in names]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise click.BadParameter(f"No vignette source for: {', '.join(missing)}")
    return paths


@beartype
def _code(nb: nbformat.NotebookNode) -> list[tuple[str, str]]:
    """Cell types and sources, the part of a notebook the `.py` determines."""
    return [(c.cell_type, c.source.strip()) for c in nb.cells]


@click.group()
def cli() -> None:
    """Execute and check the vignettes in `docs/vignettes/`."""


@cli.command()
@click.argument("names", nargs=-1)
@beartype
def execute(names: tuple[str, ...]) -> None:
    """Execute vignettes into their notebooks, all of them by default."""
    for src in _sources(names):
        nb = jupytext.read(src)
        nb.metadata["jupytext"] = {"formats": "ipynb,py:percent"}
        nb.metadata["kernelspec"] = {
            "name": "python3",
            "display_name": "Python 3",
            "language": "python",
        }
        start = time.perf_counter()
        NotebookClient(
            nb,
            timeout=CELL_TIMEOUT_S,
            kernel_name="python3",
            resources={"metadata": {"path": str(VIGNETTE_DIR)}},
        ).execute()
        jupytext.write(nb, src.with_suffix(".ipynb"))
        click.echo(f"{src.stem}: {time.perf_counter() - start:.1f} s")


@cli.command()
@beartype
def check() -> None:
    """Fail when a notebook is missing or its cells differ from the source."""
    stale = []
    for src in _sources(()):
        nb_path = src.with_suffix(".ipynb")
        if not nb_path.exists() or _code(jupytext.read(src)) != _code(
            jupytext.read(nb_path)
        ):
            stale.append(src.stem)
    if stale:
        raise click.ClickException(f"Re-execute: {', '.join(stale)}")
    click.echo("All notebooks current.")


if __name__ == "__main__":
    cli()
