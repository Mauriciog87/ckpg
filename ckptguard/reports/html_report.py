from __future__ import annotations

from collections.abc import Iterable
from importlib import resources
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from ckptguard.models import AuditReport, DiffReport, HtmlReport
from ckptguard.reports.output import write_text_atomic


def _environment() -> Environment:
    return Environment(
        loader=PackageLoader("ckptguard", "reports/templates"),
        autoescape=select_autoescape(["html", "xml", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_html_report(diff: DiffReport, audit: AuditReport, top: int = 50) -> str:
    template = _environment().get_template("report.html.j2")
    report = HtmlReport(diff=diff, audit=audit)
    return template.render(
        report=report,
        top_diffs=report.diff.tensors[:top],
        before_name=Path(report.diff.before_file.path).name,
        after_name=Path(report.diff.after_file.path).name,
    )


def write_html_report(
    diff: DiffReport,
    audit: AuditReport,
    path: Path | str,
    top: int = 50,
    protected_paths: Iterable[Path | str] = (),
) -> None:
    write_text_atomic(
        path,
        render_html_report(diff, audit, top=top),
        protected_paths=protected_paths,
    )


def template_available() -> bool:
    return resources.files("ckptguard").joinpath("reports/templates/report.html.j2").is_file()
