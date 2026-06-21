from __future__ import annotations

from collections import defaultdict

from ckptguard.models import AuditFinding, AuditReport, DiffReport, StatsReport, TensorStats

DEFAULT_FAIL_ON = ["nan", "inf", "norm-spike", "shape-drift"]
DEFAULT_NORM_SPIKE_FACTOR = 10.0
DEFAULT_SUSPICIOUS_ABS_MAX = 1_000_000.0
KNOWN_AUDIT_CATEGORIES = [
    "dtype-change",
    "inf",
    "lora-all-zero",
    "lora-dtype-mismatch",
    "lora-missing-pair",
    "lora-rank-mismatch",
    "nan",
    "norm-spike",
    "shape-drift",
    "suspicious-values",
]


def _passes(findings: list[AuditFinding], fail_on: list[str]) -> bool:
    fail_on_set = set(fail_on)
    return not any(finding.category in fail_on_set for finding in findings)


def _is_lora_tensor(name: str) -> bool:
    lowered = name.lower()
    return "lora" in lowered


def _lora_pair_key(name: str) -> tuple[str, str] | None:
    replacements = {
        "lora_a": "a",
        "lora_b": "b",
        "lora_down": "a",
        "lora_up": "b",
    }
    lowered = name.lower()
    for marker, side in replacements.items():
        if marker in lowered:
            return lowered.replace(marker, "lora_pair"), side
    return None


def _rank_for_side(tensor: TensorStats, side: str) -> int | None:
    if len(tensor.shape) < 2:
        return None
    if side == "a":
        return tensor.shape[0]
    return tensor.shape[1]


def _single_tensor_findings(tensors: list[TensorStats]) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    l2_values = sorted(tensor.l2_norm for tensor in tensors if tensor.l2_norm is not None)
    median_l2 = l2_values[len(l2_values) // 2] if l2_values else None

    for tensor in tensors:
        if tensor.nan_count > 0:
            findings.append(
                AuditFinding(
                    category="nan",
                    severity="error",
                    tensor=tensor.name,
                    message="Tensor contains NaN values.",
                    value=tensor.nan_count,
                    threshold=0,
                )
            )
        if tensor.inf_count > 0:
            findings.append(
                AuditFinding(
                    category="inf",
                    severity="error",
                    tensor=tensor.name,
                    message="Tensor contains Inf values.",
                    value=tensor.inf_count,
                    threshold=0,
                )
            )
        if (
            median_l2 is not None
            and median_l2 > 0
            and tensor.l2_norm is not None
            and tensor.l2_norm >= median_l2 * DEFAULT_NORM_SPIKE_FACTOR
        ):
            findings.append(
                AuditFinding(
                    category="norm-spike",
                    severity="error",
                    tensor=tensor.name,
                    message="Tensor L2 norm is much larger than the checkpoint median.",
                    value=tensor.l2_norm,
                    threshold=median_l2 * DEFAULT_NORM_SPIKE_FACTOR,
                )
            )
        if tensor.linf_norm is not None and tensor.linf_norm > DEFAULT_SUSPICIOUS_ABS_MAX:
            findings.append(
                AuditFinding(
                    category="suspicious-values",
                    severity="warning",
                    tensor=tensor.name,
                    message="Tensor has unusually large finite values.",
                    value=tensor.linf_norm,
                    threshold=DEFAULT_SUSPICIOUS_ABS_MAX,
                )
            )
        if _is_lora_tensor(tensor.name) and tensor.zero_ratio == 1.0:
            findings.append(
                AuditFinding(
                    category="lora-all-zero",
                    severity="warning",
                    tensor=tensor.name,
                    message="LoRA tensor is entirely zero.",
                    value=tensor.zero_ratio,
                    threshold=1.0,
                )
            )

    return findings


def _lora_findings(tensors: list[TensorStats]) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    pairs: dict[str, dict[str, TensorStats]] = defaultdict(dict)

    for tensor in tensors:
        pair = _lora_pair_key(tensor.name)
        if pair is not None:
            key, side = pair
            pairs[key][side] = tensor

    for pair in pairs.values():
        left = pair.get("a")
        right = pair.get("b")
        if left is None or right is None:
            tensor = left or right
            if tensor is not None:
                findings.append(
                    AuditFinding(
                        category="lora-missing-pair",
                        severity="warning",
                        tensor=tensor.name,
                        message="LoRA tensor does not have a matching A/B pair.",
                    )
                )
            continue

        left_rank = _rank_for_side(left, "a")
        right_rank = _rank_for_side(right, "b")
        if left_rank is not None and right_rank is not None and left_rank != right_rank:
            findings.append(
                AuditFinding(
                    category="lora-rank-mismatch",
                    severity="warning",
                    tensor=left.name,
                    message="LoRA A/B rank dimensions do not match.",
                    value=f"{left_rank}:{right_rank}",
                )
            )
        if left.dtype != right.dtype:
            findings.append(
                AuditFinding(
                    category="lora-dtype-mismatch",
                    severity="warning",
                    tensor=left.name,
                    message="LoRA A/B tensors use different dtypes.",
                    value=f"{left.dtype}:{right.dtype}",
                )
            )

    return findings


def audit_stats_report(report: StatsReport, fail_on: list[str] | None = None) -> AuditReport:
    selected_fail_on = fail_on or []
    findings = [*_single_tensor_findings(report.tensors), *_lora_findings(report.tensors)]
    return AuditReport(
        file=report.file,
        findings=findings,
        fail_on=selected_fail_on,
        passed=_passes(findings, selected_fail_on),
    )


def audit_diff_report(report: DiffReport, fail_on: list[str] | None = None) -> AuditReport:
    selected_fail_on = fail_on or []
    findings: list[AuditFinding] = []

    for diff in report.tensors:
        if "shape" in diff.changes:
            findings.append(
                AuditFinding(
                    category="shape-drift",
                    severity="error",
                    tensor=diff.name,
                    message="Tensor shape changed between checkpoints.",
                    value=str(diff.after.shape if diff.after is not None else None),
                    threshold=str(diff.before.shape if diff.before is not None else None),
                )
            )
        if "dtype" in diff.changes:
            findings.append(
                AuditFinding(
                    category="dtype-change",
                    severity="warning",
                    tensor=diff.name,
                    message="Tensor dtype changed between checkpoints.",
                    value=diff.after.dtype if diff.after is not None else None,
                    threshold=diff.before.dtype if diff.before is not None else None,
                )
            )
        if diff.before is not None and diff.after is not None:
            if diff.after.nan_count > diff.before.nan_count:
                findings.append(
                    AuditFinding(
                        category="nan",
                        severity="error",
                        tensor=diff.name,
                        message="Tensor gained NaN values.",
                        value=diff.after.nan_count - diff.before.nan_count,
                        threshold=0,
                    )
                )
            if diff.after.inf_count > diff.before.inf_count:
                findings.append(
                    AuditFinding(
                        category="inf",
                        severity="error",
                        tensor=diff.name,
                        message="Tensor gained Inf values.",
                        value=diff.after.inf_count - diff.before.inf_count,
                        threshold=0,
                    )
                )
            if (
                diff.before.l2_norm is not None
                and diff.before.l2_norm > 0
                and diff.after.l2_norm is not None
                and diff.after.l2_norm >= diff.before.l2_norm * DEFAULT_NORM_SPIKE_FACTOR
            ):
                findings.append(
                    AuditFinding(
                        category="norm-spike",
                        severity="error",
                        tensor=diff.name,
                        message="Tensor L2 norm spiked between checkpoints.",
                        value=diff.after.l2_norm,
                        threshold=diff.before.l2_norm * DEFAULT_NORM_SPIKE_FACTOR,
                    )
                )

    return AuditReport(
        file=report.after_file,
        findings=findings,
        fail_on=selected_fail_on,
        passed=_passes(findings, selected_fail_on),
    )
