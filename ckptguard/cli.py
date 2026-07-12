from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ckptguard.errors import CkptGuardError
from ckptguard.policies.audit_policy import (
    DEFAULT_BASELINE_FAIL_ON,
    DEFAULT_FAIL_ON,
    DIFFERENTIAL_AUDIT_CATEGORIES,
    KNOWN_AUDIT_CATEGORIES,
    audit_combined_report,
    audit_stats_report,
)
from ckptguard.reports.html_report import write_html_report
from ckptguard.reports.json_report import write_json_report
from ckptguard.reports.output import validate_output_paths
from ckptguard.stats.diff_stats import diff_checkpoints
from ckptguard.stats.tensor_stats import build_stats_report
from ckptguard.storage.cache_db import StatsCache

app = typer.Typer(
    help="Check local safetensors and LoRA checkpoint files.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
console = Console()
error_console = Console(stderr=True)


def _cache(cache_db: Path | None, no_cache: bool) -> StatsCache | None:
    if no_cache:
        return None
    return StatsCache(cache_db)


def _fail_on(value: str | None, has_baseline: bool) -> list[str]:
    if value is None:
        return list(DEFAULT_BASELINE_FAIL_ON if has_baseline else DEFAULT_FAIL_ON)
    if value.strip() == "":
        return []
    categories = list(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    unknown = sorted(set(categories) - set(KNOWN_AUDIT_CATEGORIES))
    if unknown:
        known = ", ".join(KNOWN_AUDIT_CATEGORIES)
        bad = ", ".join(unknown)
        raise CkptGuardError(f"Unknown --fail-on category: {bad}. Known categories: {known}")
    unavailable = sorted(set(categories) & DIFFERENTIAL_AUDIT_CATEGORIES)
    if unavailable and not has_baseline:
        names = ", ".join(unavailable)
        raise CkptGuardError(f"--fail-on category {names} requires --baseline")
    return categories


def _run(action) -> None:
    try:
        action()
    except CkptGuardError as exc:
        error_console.print(f"[red]Error:[/red] {exc}", soft_wrap=True)
        raise typer.Exit(2) from exc


def _write_json_if_requested(
    report,
    path: Path | None,
    protected_paths: Iterable[Path | str] = (),
) -> None:
    if path is not None:
        write_json_report(report, path, protected_paths=protected_paths)


def _emit_json_if_requested(
    report,
    path: Path | None,
    json_stdout: bool,
    protected_paths: Iterable[Path | str] = (),
) -> bool:
    _write_json_if_requested(report, path, protected_paths=protected_paths)
    if json_stdout:
        typer.echo(report.model_dump_json(indent=2))
    return json_stdout


def _stats_table(report) -> Table:
    table = Table(title=f"Stats: {Path(report.file.path).name}")
    table.add_column("Tensor")
    table.add_column("Shape")
    table.add_column("Dtype")
    table.add_column("Numel", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("L2", justify="right")
    table.add_column("NaN", justify="right")
    table.add_column("Inf", justify="right")

    for tensor in report.tensors:
        table.add_row(
            tensor.name,
            str(tensor.shape),
            tensor.dtype,
            str(tensor.numel),
            _format_number(tensor.mean),
            _format_number(tensor.std),
            _format_number(tensor.l2_norm),
            str(tensor.nan_count),
            str(tensor.inf_count),
        )

    return table


def _diff_table(report, top: int) -> Table:
    table = Table(title="Checkpoint diff")
    table.add_column("Tensor")
    table.add_column("Status")
    table.add_column("Changes")
    table.add_column("Score", justify="right")
    table.add_column("Cosine", justify="right")

    for diff in report.tensors[:top]:
        table.add_row(
            diff.name,
            diff.status,
            ", ".join(diff.changes) if diff.changes else "none",
            f"{diff.score:.4f}",
            _format_number(diff.cosine_distance),
        )

    return table


def _audit_table(report) -> Table:
    table = Table(title="Audit findings")
    table.add_column("Severity")
    table.add_column("Category")
    table.add_column("Tensor")
    table.add_column("Message")
    table.add_column("Value")

    for finding in report.findings:
        table.add_row(
            finding.severity,
            finding.category,
            finding.tensor or "checkpoint",
            finding.message,
            "" if finding.value is None else str(finding.value),
        )

    return table


def _format_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}"


@app.command()
def stats(
    file: Annotated[Path, typer.Argument(help="Path to a .safetensors file.")],
    json_output: Annotated[
        Path | None,
        typer.Option("--json-output", help="Write JSON report to this path."),
    ] = None,
    json_stdout: Annotated[
        bool,
        typer.Option("--json", help="Print JSON report to stdout."),
    ] = False,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Disable SQLite stats cache."),
    ] = False,
    cache_db: Annotated[
        Path | None,
        typer.Option("--cache-db", help="Path to the SQLite stats cache."),
    ] = None,
) -> None:
    def action() -> None:
        validate_output_paths([json_output], [file])
        report = build_stats_report(file, cache=_cache(cache_db, no_cache), use_cache=not no_cache)
        if not _emit_json_if_requested(
            report,
            json_output,
            json_stdout,
            protected_paths=[file],
        ):
            console.print(_stats_table(report))

    _run(action)


@app.command()
def diff(
    before: Annotated[Path, typer.Argument(help="Baseline .safetensors file.")],
    after: Annotated[Path, typer.Argument(help="Candidate .safetensors file.")],
    json_output: Annotated[
        Path | None,
        typer.Option("--json-output", help="Write JSON report to this path."),
    ] = None,
    json_stdout: Annotated[
        bool,
        typer.Option("--json", help="Print JSON report to stdout."),
    ] = False,
    top: Annotated[int, typer.Option("--top", min=1, help="Rows to show in the terminal.")] = 20,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Disable SQLite stats cache."),
    ] = False,
    cache_db: Annotated[
        Path | None,
        typer.Option("--cache-db", help="Path to the SQLite stats cache."),
    ] = None,
) -> None:
    def action() -> None:
        protected_paths = [before, after]
        validate_output_paths([json_output], protected_paths)
        report = diff_checkpoints(
            before,
            after,
            cache=_cache(cache_db, no_cache),
            use_cache=not no_cache,
        )
        if not _emit_json_if_requested(
            report,
            json_output,
            json_stdout,
            protected_paths=protected_paths,
        ):
            console.print(_diff_table(report, top))

    _run(action)


@app.command()
def audit(
    file: Annotated[Path, typer.Argument(help="Path to a .safetensors file.")],
    json_output: Annotated[
        Path | None,
        typer.Option("--json-output", help="Write JSON report to this path."),
    ] = None,
    json_stdout: Annotated[
        bool,
        typer.Option("--json", help="Print JSON report to stdout."),
    ] = False,
    fail_on: Annotated[
        str | None,
        typer.Option("--fail-on", help="Comma-separated finding categories that should fail."),
    ] = None,
    baseline: Annotated[
        Path | None,
        typer.Option("--baseline", help="Baseline .safetensors file for differential checks."),
    ] = None,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Disable SQLite stats cache."),
    ] = False,
    cache_db: Annotated[
        Path | None,
        typer.Option("--cache-db", help="Path to the SQLite stats cache."),
    ] = None,
) -> None:
    def action() -> None:
        protected_paths = [file, *([baseline] if baseline is not None else [])]
        validate_output_paths([json_output], protected_paths)
        selected_fail_on = _fail_on(fail_on, has_baseline=baseline is not None)
        cache = _cache(cache_db, no_cache)
        if baseline is None:
            stats_report = build_stats_report(
                file,
                cache=cache,
                use_cache=not no_cache,
            )
            report = audit_stats_report(stats_report, fail_on=selected_fail_on)
        else:
            diff_report = diff_checkpoints(
                baseline,
                file,
                cache=cache,
                use_cache=not no_cache,
            )
            report = audit_combined_report(diff_report, fail_on=selected_fail_on)
        if not _emit_json_if_requested(
            report,
            json_output,
            json_stdout,
            protected_paths=protected_paths,
        ):
            console.print(_audit_table(report))
        if not report.passed:
            raise typer.Exit(1)

    _run(action)


@app.command()
def report(
    before: Annotated[Path, typer.Argument(help="Baseline .safetensors file.")],
    after: Annotated[Path, typer.Argument(help="Candidate .safetensors file.")],
    html: Annotated[bool, typer.Option("--html", help="Generate a static HTML report.")] = False,
    output: Annotated[
        Path,
        typer.Option("--output", help="Path for the generated HTML report."),
    ] = Path("ckpg-report.html"),
    json_output: Annotated[
        Path | None,
        typer.Option("--json-output", help="Write diff JSON report to this path."),
    ] = None,
    top: Annotated[int, typer.Option("--top", min=1, help="Rows to include in HTML.")] = 50,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Disable SQLite stats cache."),
    ] = False,
    cache_db: Annotated[
        Path | None,
        typer.Option("--cache-db", help="Path to the SQLite stats cache."),
    ] = None,
) -> None:
    def action() -> None:
        if not html:
            raise CkptGuardError("Only --html reports are supported.")
        protected_paths = [before, after]
        validate_output_paths([output, json_output], protected_paths)
        cache = _cache(cache_db, no_cache)
        diff_report = diff_checkpoints(before, after, cache=cache, use_cache=not no_cache)
        audit_report = audit_combined_report(diff_report, fail_on=[])
        write_html_report(
            diff_report,
            audit_report,
            output,
            top=top,
            protected_paths=[*protected_paths, *([json_output] if json_output is not None else [])],
        )
        _write_json_if_requested(
            diff_report,
            json_output,
            protected_paths=[*protected_paths, output],
        )
        console.print(f"Wrote HTML report to {output}")

    _run(action)


@app.command()
def ci(
    file: Annotated[Path, typer.Argument(help="Path to a .safetensors file.")],
    fail_on: Annotated[
        str | None,
        typer.Option("--fail-on", help="Comma-separated finding categories that should fail."),
    ] = None,
    baseline: Annotated[
        Path | None,
        typer.Option("--baseline", help="Baseline .safetensors file for differential checks."),
    ] = None,
    json_output: Annotated[
        Path | None,
        typer.Option("--json-output", help="Write JSON report to this path."),
    ] = None,
    json_stdout: Annotated[
        bool,
        typer.Option("--json", help="Print JSON report to stdout."),
    ] = False,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Disable SQLite stats cache."),
    ] = False,
    cache_db: Annotated[
        Path | None,
        typer.Option("--cache-db", help="Path to the SQLite stats cache."),
    ] = None,
) -> None:
    def action() -> None:
        protected_paths = [file, *([baseline] if baseline is not None else [])]
        validate_output_paths([json_output], protected_paths)
        selected_fail_on = _fail_on(fail_on, has_baseline=baseline is not None)
        cache = _cache(cache_db, no_cache)
        if baseline is None:
            stats_report = build_stats_report(
                file,
                cache=cache,
                use_cache=not no_cache,
            )
            report = audit_stats_report(stats_report, fail_on=selected_fail_on)
        else:
            diff_report = diff_checkpoints(
                baseline,
                file,
                cache=cache,
                use_cache=not no_cache,
            )
            report = audit_combined_report(diff_report, fail_on=selected_fail_on)
        emitted_json = _emit_json_if_requested(
            report,
            json_output,
            json_stdout,
            protected_paths=protected_paths,
        )
        if report.passed:
            if not emitted_json:
                console.print("ckpg CI checks passed")
            return
        if not emitted_json:
            console.print(_audit_table(report))
        raise typer.Exit(1)

    _run(action)


def main() -> None:
    app()
