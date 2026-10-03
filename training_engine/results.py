from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


CONFIG_KEYS_FOR_TABLE = [
    "model",
    "dataset",
    "seed",
    "dims",
    "depths",
    "grid",
    "patch_size",
    "input_size",
    "batch_size",
    "epochs",
    "lr",
    "weight_decay",
    "opt",
    "mixup",
    "cutmix",
    "aa",
    "reprob",
    "smoothing",
    "num_gpus",
    "fsdp",
    "amp",
    "amp_dtype",
]


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return "(" + ",".join(str(x) for x in v) + ")"
    return str(v)


def _slug(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower())
    return text.strip("-") or "run"


def run_report_stem(cfg: dict[str, Any], when: datetime | None = None) -> str:
    when = when or datetime.now(timezone.utc)
    return (
        f"{_slug(cfg.get('model', 'model'))}_"
        f"{_slug(cfg.get('dataset', 'data'))}_"
        f"{when.strftime('%Y%m%d_%H%M%S')}"
    )


def _report_markdown(
    cfg: dict[str, Any],
    metrics: dict[str, Any],
    n_parameters: int | None,
    stamp: datetime,
    report_name: str,
) -> str:
    params_m = n_parameters / 1e6 if n_parameters is not None else metrics.get("params_m")
    time_h = metrics.get("train_time_h")
    time_s = metrics.get("train_time_s")
    if time_s is None and time_h is not None:
        time_s = float(time_h) * 3600.0
    time_cell = "—"
    if time_h is not None:
        time_cell = f"{_fmt(time_h)} h"
        if time_s is not None:
            time_cell += f" ({_fmt(time_s)} s)"

    model = cfg.get("model", "—")
    dataset = cfg.get("dataset", "—")
    lines = [
        f"# {model} · {dataset}",
        "",
        f"- File: `{report_name}`",
        f"- Finished: `{stamp.strftime('%Y-%m-%d %H:%M:%S UTC')}`",
        f"- Seed: `{_fmt(cfg.get('seed'))}`",
        "",
        "## Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Model | `{model}` |",
        f"| Dataset | `{dataset}` |",
        f"| Params | {_fmt(params_m)} M |",
        f"| Peak memory | {_fmt(metrics.get('peak_mem_gb'))} GiB |",
        f"| Train time | {time_cell} |",
        f"| Epochs | {_fmt(metrics.get('epochs'))} |",
        f"| Best epoch | {_fmt(metrics.get('best_epoch'))} |",
        f"| GPUs | {_fmt(cfg.get('num_gpus'))} |",
    ]
    model_name = str(model)
    if model_name.startswith("LKAT"):
        lines.append("| Normalization | `Post-LayerNorm + σ(x) residual + QK-RMSNorm + RoPE` |")
    elif model_name.startswith("KATT"):
        lines.append("| Normalization | `RMSNorm + QK-RMSNorm + RoPE` |")
    lines.extend(
        [
        "",
        "## Accuracy",
        "",
        "| | Acc@1 (%) | Error (%) | Acc@5 (%) |",
        "|---|---:|---:|---:|",
        f"| Best | {_fmt(metrics.get('best_acc1'))} | {_fmt(metrics.get('best_error'))} | — |",
        f"| Final | {_fmt(metrics.get('final_acc1'))} | {_fmt(metrics.get('final_error'))} | {_fmt(metrics.get('final_acc5'))} |",
        "",
        "Error = 100 − Acc@1. Peak memory is `torch.cuda.max_memory_allocated`.",
        "",
        ]
    )
    if model_name.startswith("LKAT") or model_name.startswith("KATT"):
        block_norm = (
            "Post-LayerNorm + σ(x) residual"
            if model_name.startswith("LKAT")
            else "RMSNorm"
        )
        lines.extend(
            [
                "## Normalization",
                "",
                "| Location | Technique |",
                "|---|---|",
                f"| Block (post-attn / post-MLP) | {block_norm} |" if model_name.startswith("LKAT") else f"| Block (pre-attn / pre-MLP) | {block_norm} |",
                f"| Final | {'LayerNorm' if model_name.startswith('LKAT') else block_norm} |",
                "| Attention Q/K | QK-RMSNorm |",
            ]
        )
        if model_name.startswith("LKAT"):
            lines.append("| GLA output path | LayerNorm (replaces FLA output RMSNorm) |")
        lines.extend(
            [
                f"| Position | Abs pos + 2D RoPE (`{_fmt(cfg.get('rope_mode', '2dv1'))}`) |",
                "",
                (
                    "Stack: post-LayerNorm blocks with σ(x)⊙x gated residuals; RMSNorm on attention Q/K."
                    if model_name.startswith("LKAT")
                    else f"Stack: {block_norm} on the residual stream; RMSNorm on attention Q/K."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Train config",
            "",
            "| Key | Value |",
            "|---|---|",
        ]
    )
    for key in CONFIG_KEYS_FOR_TABLE:
        if key in cfg:
            lines.append(f"| `{key}` | `{_fmt(cfg[key])}` |")
    for key in (
        "aug",
        "mixup_prob",
        "mixup_switch_prob",
        "mixup_mode",
        "warmup_epochs",
        "min_lr",
        "data_path",
        "output_dir",
        "embed_dim",
        "depth",
        "num_heads",
        "num_registers",
        "use_rope",
        "rope_mode",
    ):
        if key in cfg and key not in CONFIG_KEYS_FOR_TABLE:
            lines.append(f"| `{key}` | `{_fmt(cfg[key])}` |")
    lines.append("")
    return "\n".join(lines)


def save_resolved_config(output_dir: str | Path, cfg: dict[str, Any]) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config_resolved.yaml"
    # drop non-serializable objects
    clean = {k: v for k, v in cfg.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
    with path.open("w") as f:
        yaml.safe_dump(clean, f, sort_keys=False)
    return path


def write_run_results_md(
    output_dir: str | Path,
    cfg: dict[str, Any],
    metrics: dict[str, Any],
    n_parameters: int | None = None,
    repo_root: str | Path | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc)
    stem = run_report_stem(cfg, stamp)
    name = f"{stem}.md"
    body = _report_markdown(cfg, metrics, n_parameters, stamp, name)

    dated = output_dir / name
    dated.write_text(body)
    (output_dir / "results.md").write_text(body)

    if repo_root is not None:
        archive = Path(repo_root) / "results"
        archive.mkdir(parents=True, exist_ok=True)
        (archive / name).write_text(body)
        return archive / name
    return dated


_RUN_LOG_COLS = (
    "| Model | Dataset | Acc@1 | Epochs | GPUs | Peak GiB | Time (h) | Notes |"
)
_RUN_LOG_SEP = "|---|---|---:|---:|---:|---:|---:|---|"


def _default_notes(cfg: dict[str, Any], metrics: dict[str, Any]) -> str:
    note = metrics.get("notes") or cfg.get("notes")
    if note:
        return str(note)
    model = str(cfg.get("model") or "")
    if model.startswith("LKAT"):
        return "Post-LayerNorm + σ(x) residual + QK-RMSNorm + RoPE"
    if model.startswith("KATT"):
        return "RMSNorm + QK-RMSNorm + RoPE"
    return "—"


def _run_log_row(
    cfg: dict[str, Any],
    metrics: dict[str, Any],
    n_parameters: int | None,
) -> str:
    del n_parameters  # kept for call-site compatibility
    peak = metrics.get("peak_mem_gb")
    time_h = metrics.get("train_time_h")
    if time_h is None and metrics.get("train_time_s") is not None:
        time_h = float(metrics["train_time_s"]) / 3600.0
    return (
        f"| {cfg.get('model')} "
        f"| {cfg.get('dataset')} "
        f"| {_fmt(metrics.get('best_acc1'))} "
        f"| {_fmt(metrics.get('epochs'))} "
        f"| {_fmt(cfg.get('num_gpus'))} "
        f"| {_fmt(peak)} "
        f"| {_fmt(time_h)} "
        f"| {_default_notes(cfg, metrics)} |"
    )


def _dataset_section(dataset: Any) -> str:
    ds = str(dataset or "").strip().lower()
    if ds == "imagenet100":
        return "### ImageNet-100"
    return "### CIFAR-10 / CIFAR-100"


def _insert_row_into_section(text: str, section: str, row: str) -> str:
    """Append ``row`` after the last data row of ``section`` (creates section if missing)."""
    lines = text.splitlines()
    try:
        sec_i = next(i for i, l in enumerate(lines) if l.strip() == section)
    except StopIteration:
        if not text.endswith("\n"):
            text += "\n"
        return text + f"\n{section}\n\n{_RUN_LOG_COLS}\n{_RUN_LOG_SEP}\n{row}\n"

    header_i = None
    for i in range(sec_i + 1, len(lines)):
        if lines[i].startswith("### "):
            break
        if lines[i].startswith("| Model |"):
            header_i = i
            break
    if header_i is None:
        insert_at = sec_i + 1
        block = ["", _RUN_LOG_COLS, _RUN_LOG_SEP, row]
        lines[insert_at:insert_at] = block
        return "\n".join(lines) + ("\n" if text.endswith("\n") else "")

    j = header_i + 1
    if j < len(lines) and lines[j].startswith("|---"):
        j += 1
    last_row = j - 1
    while j < len(lines) and lines[j].startswith("|"):
        last_row = j
        j += 1
    lines.insert(last_row + 1, row)
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def append_aggregate_results(
    repo_root: str | Path,
    cfg: dict[str, Any],
    metrics: dict[str, Any],
    n_parameters: int | None = None,
) -> Path:
    repo_root = Path(repo_root)
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "RESULTS.md"

    section = _dataset_section(cfg.get("dataset"))
    row = _run_log_row(cfg, metrics, n_parameters)

    if not path.exists():
        bootstrap = [
            "# helloKAN results",
            "",
            "## Run log",
            "",
            "Trainer appends a row after each finished run. Acc@1 is **best** validation Top-1. "
            "Peak GiB is `torch.cuda.max_memory_allocated`. Time is wall-clock hours.",
            "",
            "### CIFAR-10 / CIFAR-100",
            "",
            _RUN_LOG_COLS,
            _RUN_LOG_SEP,
            "",
            "### ImageNet-100",
            "",
            _RUN_LOG_COLS,
            _RUN_LOG_SEP,
            "",
        ]
        path.write_text("\n".join(bootstrap))

    path.write_text(_insert_row_into_section(path.read_text(), section, row))

    jsonl = results_dir / "results.jsonl"
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {k: cfg.get(k) for k in CONFIG_KEYS_FOR_TABLE},
        "metrics": metrics,
        "params": n_parameters,
        "report": metrics.get("report_file"),
        "notes": metrics.get("notes") or cfg.get("notes"),
    }
    with jsonl.open("a") as f:
        f.write(json.dumps(record) + "\n")

    return path


def write_epoch_table_md(output_dir: str | Path, history: list[dict[str, Any]]) -> Path:
    output_dir = Path(output_dir)
    path = output_dir / "epochs.md"
    lines = [
        "# Epoch log",
        "",
        "| Epoch | Train Loss | LR | Acc@1 | Acc@5 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for h in history:
        lines.append(
            f"| {h.get('epoch')} | {_fmt(h.get('train_loss'))} | {_fmt(h.get('lr'))} | "
            f"{_fmt(h.get('test_acc1'))} | {_fmt(h.get('test_acc5'))} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))
    return path
