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
uv run ckpg audit adapter_model.safetensors --fail-on nan,inf,norm-spike,shape-drift
uv run ckpg diff before.safetensors after.safetensors
uv run ckpg report before.safetensors after.safetensors --html
uv run ckpg ci adapter_model.safetensors --fail-on nan,inf,norm-spike,shape-drift
```

Use `--json` to print machine-readable output to stdout, or `--json-output PATH` to write it to a file.

## What ckpg checks

- Tensor shape, dtype, element count, min, max, mean, std, L2 norm, Linf norm, zero ratio, NaN count, Inf count, and hash.
- Checkpoint diffs for added, removed, changed, shape drift, dtype changes, distribution shifts, and norm deltas.
- Audit findings for NaN, Inf, norm spikes, shape drift, dtype changes, suspicious values, and LoRA-specific anomalies.

## Boundaries

ckpg only supports `safetensors` in this version.

It is not a training-time debugger, tensor database, vector database, model observability platform, cloud dashboard, or MLOps platform.
