from __future__ import annotations

from importlib import resources
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from ckptguard.models import AuditReport, DiffReport, HtmlReport


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
    return template.render(report=report, top_diffs=report.diff.tensors[:top])


def write_html_report(
    diff: DiffReport,
    audit: AuditReport,
    path: Path | str,
    top: int = 50,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html_report(diff, audit, top=top), encoding="utf-8")


def template_available() -> bool:
    return resources.files("ckptguard").joinpath("reports/templates/report.html.j2").is_file()
