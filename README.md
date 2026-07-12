# ckpg

ckpg checks local `safetensors` and LoRA checkpoint files before they move into CI, review, or release workflows.

It reads local files, reports tensor statistics, compares checkpoints, audits common failure signals, and writes JSON output for automation.

## Install

For local development:

```powershell
uv sync --dev
uv run ckpg --help
```

## Usage

```powershell
uv run ckpg stats model.safetensors
uv run ckpg audit adapter_model.safetensors
uv run ckpg audit candidate.safetensors --baseline baseline.safetensors
uv run ckpg diff before.safetensors after.safetensors
uv run ckpg report before.safetensors after.safetensors --html
uv run ckpg ci candidate.safetensors --baseline baseline.safetensors
```

Use `--json` to print machine-readable output to stdout, or `--json-output PATH` to write it to a file.
Errors are written to stderr so stdout remains safe for JSON, tables, and successful results.

Without `--baseline`, `audit` and `ci` fail on `nan`, `inf`, and `norm-spike` by default.
With `--baseline`, they also fail on `shape-drift`, including added, removed, and reshaped
tensors. Differential categories such as `shape-drift` and `dtype-change` require a baseline.

## What ckpg checks

- Tensor shape, dtype, element count, min, max, mean, std, L2 norm, Linf norm, zero ratio, NaN count, Inf count, and hash.
- Checkpoint diffs for added, removed, changed, shape drift, dtype changes, distribution shifts, and norm deltas.
- Audit findings for NaN, Inf, norm spikes, shape drift, dtype changes, suspicious values, and LoRA-specific anomalies.

The stats cache is enabled by default and validates each checkpoint with a full-file SHA-256 before
reusing results. This adds one sequential read per cached invocation; use `--no-cache` when that cost
outweighs reuse.

## Boundaries

ckpg only supports `safetensors` in this version. Standard NumPy dtypes and BF16 are supported. FP8
is rejected with an explicit error because the NumPy safetensors backend cannot materialize it.

JSON reports retain absolute checkpoint paths for automation. HTML reports show checkpoint file names
only, so local directory paths are not exposed in shared reports.

It is not a training-time debugger, tensor database, vector database, model observability platform, cloud dashboard, or MLOps platform.
