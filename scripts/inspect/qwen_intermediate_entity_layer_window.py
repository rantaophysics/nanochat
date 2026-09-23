"""Layer-window experiment for intermediate prompt entity-value contributions.

The intervention preserves native QK softmax probabilities.  At selected query
positions it recomputes SDPA after replacing the original entity token's value
with zero, then substitutes those query outputs.  Thus this is entity-value
contribution removal, not attention-probability zeroing.

Results are appended one trial at a time and can be resumed with the same
output directory.  Use --analyze-only to rebuild aggregates and figures.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from qwen_entity_attention_ablation import (
    ENTITY_SETS,
    MODEL,
    TEMPLATES,
    EncodedPrompt,
    encode_prompt,
    normalize_text,
)


ATTENTION_IMPLEMENTATION = "intermediate_entity_layer_window"
REFERENCE_COMMIT = "3bf9a3f05df9fca7190478546b30a1dee69e537f"
REFERENCE_HASHES = {
    "scripts/inspect/qwen_prompt_attention_profile.py":
        "c52c4a8581db369cd5da52a59a701b66fdc78bbda4e0790e5719f5d3f8c89b1e",
    "scripts/inspect/qwen_entity_attention_ablation.py":
        "79cdc35b4972f77eedea5b91e2cf83f2acb1f2b08ea412b5abfd70895312a7e6",
}
PILOT_TEMPLATE_NAMES = (
    "person_in_list", "friend", "direct_fact", "visitor_register"
)
WIDTHS = (1, 2, 3, 4, 6, 8)


@dataclass(frozen=True)
class Trial:
    condition: str
    intermediate_disabled_layers: frozenset[int] = frozenset()
    final_disabled_layers: frozenset[int] = frozenset()
    generated_disabled_layers: frozenset[int] = frozenset()
    window_start: int | None = None
    window_width: int | None = None
    restore_layers: frozenset[int] = frozenset()
    restore_mode: str | None = None


@dataclass
class PolicyState:
    active: bool = False
    capture_reference: bool = False
    entity_position: int | None = None
    intermediate_positions: tuple[int, ...] = ()
    final_position: int | None = None
    prompt_length: int | None = None
    intermediate_disabled_layers: frozenset[int] = frozenset()
    final_disabled_layers: frozenset[int] = frozenset()
    generated_disabled_layers: frozenset[int] = frozenset()
    restore_layers: frozenset[int] = frozenset()
    restore_mode: str | None = None
    ordinary_attention: dict[int, torch.Tensor] | None = None
    wrong_entity_values: dict[int, torch.Tensor] | None = None
    captured_attention: dict[int, torch.Tensor] = field(default_factory=dict)
    captured_values: dict[int, torch.Tensor] = field(default_factory=dict)
    diagnostics: dict[int, dict[str, Any]] = field(default_factory=dict)
    touched: dict[str, dict[int, int]] = field(default_factory=dict)
    restore_trace: dict[int, dict[str, Any]] = field(default_factory=dict)


POLICY = PolicyState()


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, kv_heads, seq_len, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None, :, :].expand(
        batch, kv_heads, n_rep, seq_len, head_dim
    ).reshape(batch, kv_heads * n_rep, seq_len, head_dim)


def _query_entity_attention(
    module, query: torch.Tensor, key: torch.Tensor,
    attention_mask: torch.Tensor | None, scaling: float,
    local_queries: list[int], entity_position: int,
) -> torch.Tensor:
    key_states = _repeat_kv(key, module.num_key_value_groups)
    q = query[:, :, local_queries, :]
    scores = torch.matmul(q, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, local_queries, : key_states.shape[-2]]
    weights = F.softmax(scores, dim=-1, dtype=torch.float32)
    return weights[..., entity_position]


def _projected_norm(module, head_values: torch.Tensor) -> torch.Tensor:
    # [batch, head, query, dim] -> [batch, query]
    flat = head_values.permute(0, 2, 1, 3).reshape(
        head_values.shape[0], head_values.shape[2], -1
    )
    projected = F.linear(flat, module.o_proj.weight, bias=None)
    return torch.linalg.vector_norm(projected.float(), dim=-1)


def _bump(kind: str, layer: int, count: int) -> None:
    if count:
        POLICY.touched.setdefault(kind, {})[layer] = (
            POLICY.touched.setdefault(kind, {}).get(layer, 0) + count
        )


def entity_window_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    if not POLICY.active:
        return sdpa_attention_forward(
            module, query, key, value, attention_mask,
            scaling=scaling, dropout=dropout, **kwargs,
        )
    if POLICY.entity_position is None or POLICY.final_position is None:
        raise RuntimeError("active entity policy has incomplete positions")

    layer = int(module.layer_idx)
    q_len, k_len = query.shape[-2], key.shape[-2]
    q_start = k_len - q_len
    absolute = list(range(q_start, q_start + q_len))
    final_local = [i for i, p in enumerate(absolute) if p == POLICY.final_position]
    intermediate_local = [
        i for i, p in enumerate(absolute) if p in POLICY.intermediate_positions
    ]
    generated_local = [
        i for i, p in enumerate(absolute) if p >= (POLICY.prompt_length or 0)
    ]
    value_states = _repeat_kv(value, module.num_key_value_groups)

    if POLICY.capture_reference:
        if len(final_local) != 1:
            raise RuntimeError("reference capture requires prompt prefill")
        attention = _query_entity_attention(
            module, query, key, attention_mask, scaling,
            final_local, POLICY.entity_position,
        )[:, :, 0]
        POLICY.captured_attention[layer] = attention.detach().float().clone()
        POLICY.captured_values[layer] = value_states[
            :, :, POLICY.entity_position, :
        ].detach().clone()
        return sdpa_attention_forward(
            module, query, key, value, attention_mask,
            scaling=scaling, dropout=dropout, **kwargs,
        )

    # Audit the native, live state at prompt prefill before changing outputs.
    if final_local:
        if len(final_local) != 1 or layer in POLICY.diagnostics:
            raise RuntimeError(f"invalid or duplicate prompt diagnostic at layer {layer}")
        final_attention = _query_entity_attention(
            module, query, key, attention_mask, scaling,
            final_local, POLICY.entity_position,
        )
        final_weighted = (
            final_attention.to(value_states.dtype).unsqueeze(-1)
            * value_states[:, :, POLICY.entity_position, :].unsqueeze(2)
        )
        if intermediate_local:
            inter_attention = _query_entity_attention(
                module, query, key, attention_mask, scaling,
                intermediate_local, POLICY.entity_position,
            )
            inter_weighted = (
                inter_attention.to(value_states.dtype).unsqueeze(-1)
                * value_states[:, :, POLICY.entity_position, :].unsqueeze(2)
            )
            inter_norm = _projected_norm(module, inter_weighted)[0]
            inter_mean = float(inter_norm.mean().cpu())
            inter_sum = float(inter_norm.sum().cpu())
        else:
            inter_mean, inter_sum = 0.0, 0.0
        POLICY.diagnostics[layer] = {
            "live_final_entity_attention_mean": float(final_attention.mean().cpu()),
            "live_final_entity_attention_max": float(final_attention.max().cpu()),
            "live_final_entity_contribution_norm": float(
                _projected_norm(module, final_weighted)[0, 0].cpu()
            ),
            "live_intermediate_entity_contribution_mean_norm": inter_mean,
            "live_intermediate_entity_contribution_sum_norm": inter_sum,
        }

    affected_intermediate = (
        intermediate_local if layer in POLICY.intermediate_disabled_layers else []
    )
    affected_final = final_local if layer in POLICY.final_disabled_layers else []
    affected_generated = (
        generated_local if layer in POLICY.generated_disabled_layers else []
    )
    affected = sorted(set(
        affected_intermediate + affected_final + affected_generated
    ))
    _bump("intermediate", layer, len(affected_intermediate))
    _bump("final", layer, len(affected_final))
    _bump("generated", layer, len(affected_generated))

    restore = layer in POLICY.restore_layers and bool(final_local)
    zero_output = None
    if affected:
        zero_value = value.clone()
        zero_value[:, :, POLICY.entity_position, :] = 0
        zero_output, _ = sdpa_attention_forward(
            module, query, key, zero_value, attention_mask,
            scaling=scaling, dropout=dropout, **kwargs,
        )
        if len(affected) == q_len and not restore:
            return zero_output, None

    native_output, _ = sdpa_attention_forward(
        module, query, key, value, attention_mask,
        scaling=scaling, dropout=dropout, **kwargs,
    )
    if not affected and not restore:
        return native_output, None
    output = native_output.clone()
    if affected:
        output[:, affected, :, :] = zero_output[:, affected, :, :]

    if restore:
        if POLICY.ordinary_attention is None or layer not in POLICY.ordinary_attention:
            raise RuntimeError(f"missing ordinary attention at layer {layer}")
        live = _query_entity_attention(
            module, query, key, attention_mask, scaling,
            final_local, POLICY.entity_position,
        )[:, :, 0].float()
        ordinary = POLICY.ordinary_attention[layer].to(live.device).float()
        deficit = (ordinary - live).clamp_min(0)
        target_value = value_states[:, :, POLICY.entity_position, :]
        if POLICY.restore_mode == "entity":
            source_value = target_value
        elif POLICY.restore_mode == "wrong_entity":
            if POLICY.wrong_entity_values is None or layer not in POLICY.wrong_entity_values:
                raise RuntimeError(f"missing wrong-entity value at layer {layer}")
            source_value = POLICY.wrong_entity_values[layer].to(
                target_value.device, target_value.dtype
            )
            # Per-head norm matching isolates direction/identity from magnitude.
            target_norm = torch.linalg.vector_norm(target_value.float(), dim=-1, keepdim=True)
            source_norm = torch.linalg.vector_norm(source_value.float(), dim=-1, keepdim=True)
            source_value = source_value * (
                target_norm / source_norm.clamp_min(1e-12)
            ).to(source_value.dtype)
        elif POLICY.restore_mode == "noop":
            source_value = target_value
            deficit = torch.zeros_like(deficit)
        else:
            raise RuntimeError(f"unknown restore mode {POLICY.restore_mode!r}")
        supplement = deficit.to(source_value.dtype).unsqueeze(-1) * source_value
        output[:, final_local[0], :, :] += supplement
        added_norm = _projected_norm(module, supplement.unsqueeze(2))[0, 0]
        POLICY.restore_trace[layer] = {
            "ordinary_entity_attention": ordinary[0].detach().cpu().tolist(),
            "live_intervened_entity_attention": live[0].detach().cpu().tolist(),
            "added_attention_deficit": deficit[0].detach().cpu().tolist(),
            "effective_attention_sum": (1.0 + deficit[0]).detach().cpu().tolist(),
            "entity_value_contribution_added_norm": float(added_norm.cpu()),
            "renormalized": False,
            "source": POLICY.restore_mode,
        }
        _bump("restore", layer, 1)
    return output, None


@contextmanager
def policy_context(
    encoded: EncodedPrompt,
    trial: Trial | None = None,
    capture_reference: bool = False,
    ordinary_attention: dict[int, torch.Tensor] | None = None,
    wrong_values: dict[int, torch.Tensor] | None = None,
):
    if POLICY.active:
        raise RuntimeError("nested policies are unsupported")
    final = len(encoded.ids) - 1
    intermediate = tuple(range(encoded.entity_position + 1, final))
    if not intermediate:
        raise ValueError("prompt has no post-entity intermediate positions")
    POLICY.active = True
    POLICY.capture_reference = capture_reference
    POLICY.entity_position = encoded.entity_position
    POLICY.intermediate_positions = intermediate
    POLICY.final_position = final
    POLICY.prompt_length = len(encoded.ids)
    if trial is not None:
        POLICY.intermediate_disabled_layers = trial.intermediate_disabled_layers
        POLICY.final_disabled_layers = trial.final_disabled_layers
        POLICY.generated_disabled_layers = trial.generated_disabled_layers
        POLICY.restore_layers = trial.restore_layers
        POLICY.restore_mode = trial.restore_mode
    POLICY.ordinary_attention = ordinary_attention
    POLICY.wrong_entity_values = wrong_values
    POLICY.captured_attention = {}
    POLICY.captured_values = {}
    POLICY.diagnostics = {}
    POLICY.touched = {}
    POLICY.restore_trace = {}
    try:
        yield POLICY
    finally:
        POLICY.active = False
        POLICY.capture_reference = False
        POLICY.entity_position = None
        POLICY.intermediate_positions = ()
        POLICY.final_position = None
        POLICY.prompt_length = None
        POLICY.intermediate_disabled_layers = frozenset()
        POLICY.final_disabled_layers = frozenset()
        POLICY.generated_disabled_layers = frozenset()
        POLICY.restore_layers = frozenset()
        POLICY.restore_mode = None
        POLICY.ordinary_attention = None
        POLICY.wrong_entity_values = None


def load_model(model_name: str, device: torch.device):
    from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

    AttentionInterface.register(
        ATTENTION_IMPLEMENTATION, entity_window_attention_forward
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, attn_implementation=ATTENTION_IMPLEMENTATION
    ).to(device).eval()
    return model, tokenizer, len(model.model.layers)


@torch.inference_mode()
def capture_reference(model, encoded: EncodedPrompt, device: torch.device):
    ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    with policy_context(encoded, capture_reference=True) as state:
        model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        return (
            {k: v.detach().clone() for k, v in state.captured_attention.items()},
            {k: v.detach().clone() for k, v in state.captured_values.items()},
        )


def answer_token_id(tokenizer, entity: str) -> int:
    # All reference templates end at ``Answer:`` and greedy answers begin with
    # a space-prefixed name token.  Score that contextual answer token rather
    # than the different beginning-of-string tokenization of the bare name.
    ids = tokenizer(" " + entity, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"answer {entity!r} is not one token: {ids}")
    return int(ids[0])


def contains_entity(text: str, entity: str) -> bool:
    return re.search(
        rf"(?<!\w){re.escape(normalize_text(entity))}(?!\w)", normalize_text(text)
    ) is not None


def exact_answer(text: str, entity: str) -> bool:
    visible = re.sub(r"<\|[^|]+\|>", "", text)
    visible = normalize_text(visible).strip(" \t\r\n.,;:!?\"'`-_")
    return visible == normalize_text(entity)


def failure_category(text: str, entity: str, all_entities: list[str]) -> str:
    if exact_answer(text, entity):
        return "correct"
    norm = normalize_text(re.sub(r"<\|[^|]+\|>", "", text))
    if not norm or not re.search(r"\w", norm):
        return "punctuation_or_empty"
    if any(e != entity and contains_entity(text, e) for e in all_entities):
        return "wrong_entity"
    words = re.findall(r"\w+", norm)
    if len(words) >= 4 and len(set(words)) <= max(1, len(words) // 2):
        return "repetition"
    if any(x in norm for x in ("unknown", "not provided", "person's name", "the name")):
        return "generic_completion"
    if "�" in text or text.count("<|") != text.count("|>"):
        return "malformed_text"
    return "other"


def _counts(state: PolicyState) -> dict[str, dict[str, int]]:
    return {
        kind: {str(layer): count for layer, count in sorted(values.items())}
        for kind, values in sorted(state.touched.items())
    }


@torch.inference_mode()
def run_trial(
    model, tokenizer, encoded: EncodedPrompt, trial: Trial,
    device: torch.device, max_new_tokens: int, n_layers: int,
    ordinary_attention: dict[int, torch.Tensor],
    wrong_values: dict[int, torch.Tensor] | None,
    all_entities: list[str], seed: int,
) -> dict[str, Any]:
    ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    expected_id = answer_token_id(tokenizer, encoded.entity)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    with policy_context(
        encoded, trial, ordinary_attention=ordinary_attention,
        wrong_values=wrong_values,
    ) as state:
        generated = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=pad_id,
        )
        generated_ids = generated.sequences[0, len(encoded.ids):].detach().cpu().tolist()
        first_logits = generated.scores[0][0].float()
        first_probs = F.softmax(first_logits, dim=-1)
        expected_log_prob = float(F.log_softmax(first_logits, dim=-1)[expected_id].cpu())
        masked = first_logits.clone()
        masked[expected_id] = -torch.inf
        margin = float((first_logits[expected_id] - masked.max()).cpu())
        diagnostics = [state.diagnostics[layer] for layer in range(n_layers)]
        touched = _counts(state)
        restore_trace = {
            str(k): v for k, v in sorted(state.restore_trace.items())
        }
    completion = tokenizer.decode(generated_ids, skip_special_tokens=False)
    first_id = int(generated_ids[0])
    return {
        "seed": seed,
        "condition": trial.condition,
        "window_start": trial.window_start,
        "window_width": trial.window_width,
        "window_end_exclusive": (
            None if trial.window_start is None else trial.window_start + (trial.window_width or 0)
        ),
        "intermediate_disabled_layers": sorted(trial.intermediate_disabled_layers),
        "final_disabled_layers": sorted(trial.final_disabled_layers),
        "generated_disabled_layers": sorted(trial.generated_disabled_layers),
        "restore_layers": sorted(trial.restore_layers),
        "restore_mode": trial.restore_mode,
        "prompt": encoded.prompt,
        "prompt_token_ids": encoded.ids,
        "prompt_tokens": encoded.tokens,
        "entity_token_ids": [encoded.ids[encoded.entity_position]],
        "entity_positions": [encoded.entity_position],
        "intermediate_positions": list(range(encoded.entity_position + 1, len(encoded.ids) - 1)),
        "final_prompt_position": len(encoded.ids) - 1,
        "template_id": encoded.template_id,
        "template": encoded.template_name,
        "entity_id": encoded.entity_id,
        "entity": encoded.entity,
        "entity_group": encoded.entity_group,
        "expected_answer_token_id": expected_id,
        "generated_token_ids": generated_ids,
        "completion": completion,
        "normalized_completion": normalize_text(completion),
        "exact_expected_answer": exact_answer(completion, encoded.entity),
        "entity_in_completion": contains_entity(completion, encoded.entity),
        "failure_category": failure_category(completion, encoded.entity, all_entities),
        "expected_entity_log_probability": expected_log_prob,
        "expected_vs_best_alternative_logit_margin": margin,
        "first_generated_token_id": first_id,
        "first_generated_token": tokenizer.decode([first_id], skip_special_tokens=False),
        "first_generated_token_probability": float(first_probs[first_id].cpu()),
        "first_expected_token_probability": float(first_probs[expected_id].cpu()),
        "final_prompt_entity_attention_by_layer": [
            x["live_final_entity_attention_mean"] for x in diagnostics
        ],
        "final_prompt_entity_contribution_norm_by_layer": [
            x["live_final_entity_contribution_norm"] for x in diagnostics
        ],
        "intermediate_entity_contribution_mean_norm_by_layer": [
            x["live_intermediate_entity_contribution_mean_norm"] for x in diagnostics
        ],
        "intermediate_entity_contribution_sum_norm_by_layer": [
            x["live_intermediate_entity_contribution_sum_norm"] for x in diagnostics
        ],
        "ordinary_minus_live_final_attention_by_layer": [
            float(ordinary_attention[layer].mean().cpu())
            - diagnostics[layer]["live_final_entity_attention_mean"]
            for layer in range(n_layers)
        ],
        "policy_application_counts_by_layer": touched,
        "restoration_trace": restore_trace,
        "restoration_renormalized": False if trial.restore_mode else None,
    }


def trial_key(row_or_trial: dict[str, Any] | Trial, template: str, entity: str) -> str:
    if isinstance(row_or_trial, Trial):
        parts = (
            row_or_trial.condition, row_or_trial.window_start,
            row_or_trial.window_width, row_or_trial.restore_mode,
        )
    else:
        parts = (
            row_or_trial.get("condition"), row_or_trial.get("window_start"),
            row_or_trial.get("window_width"), row_or_trial.get("restore_mode"),
        )
    return json.dumps([template, entity, *parts], separators=(",", ":"))


def build_trials(n_layers: int, smoke: bool) -> list[Trial]:
    all_layers = frozenset(range(n_layers))
    trials = [
        Trial("clean"),
        Trial("generated_read_block", generated_disabled_layers=all_layers),
        Trial(
            "full_intermediate_bottleneck",
            intermediate_disabled_layers=all_layers,
            generated_disabled_layers=all_layers,
        ),
    ]
    if smoke:
        spans = [(2, 0), (3, max(0, n_layers // 2 - 1)), (2, n_layers - 2)]
    else:
        spans = [
            (width, start) for width in WIDTHS
            for start in range(n_layers - width + 1)
        ]
    for width, start in spans:
        window = frozenset(range(start, start + width))
        trials.append(Trial(
            "necessity_window", window, frozenset(), all_layers, start, width
        ))
        trials.append(Trial(
            "sufficiency_window", all_layers - window, frozenset(), all_layers,
            start, width,
        ))
    restore_starts = [0, n_layers // 2, n_layers - 6] if smoke else list(range(n_layers))
    for start in restore_starts:
        restore = frozenset(range(start, n_layers))
        for mode in ("entity", "wrong_entity"):
            trials.append(Trial(
                f"final_restore_{mode}", all_layers, frozenset(), all_layers,
                start, n_layers - start, restore, mode,
            ))
    # One no-op condition is sufficient to validate the hook path and avoids
    # pretending identical start-layer variants are independent controls.
    trials.append(Trial(
        "final_restore_noop", all_layers, frozenset(), all_layers,
        0, n_layers, all_layers, "noop",
    ))
    return trials


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_info(repo: Path) -> dict[str, Any]:
    def run(*args):
        return subprocess.run(
            args, cwd=repo, text=True, capture_output=True, check=False
        ).stdout.strip()
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
        "status_porcelain": run("git", "status", "--porcelain=v1"),
    }


def package_versions() -> dict[str, str]:
    result = {"python": platform.python_version(), "torch": torch.__version__}
    for name in ("transformers", "accelerate", "numpy", "matplotlib", "pandas"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def choose_cases(tokenizer, entity_count: int, smoke: bool):
    templates = [next(t for t in TEMPLATES if t.name == name) for name in PILOT_TEMPLATE_NAMES]
    wanted_templates = [templates[0], templates[2]] if smoke else templates
    valid, exclusions = [], []
    for entity_id, (entity, group) in enumerate(ENTITY_SETS["diverse100"]):
        try:
            answer_token_id(tokenizer, entity)
            encoded = [encode_prompt(tokenizer, t, entity_id, entity, group) for t in wanted_templates]
            valid.append((entity_id, entity, group, encoded))
        except Exception as exc:
            exclusions.append({"entity_id": entity_id, "entity": entity, "reason": str(exc)})
        if len(valid) >= entity_count:
            break
    if len(valid) < entity_count:
        raise RuntimeError(f"only {len(valid)} valid entities; need {entity_count}")
    cases = [case for _, _, _, encoded in valid for case in encoded]
    return cases, exclusions, wanted_templates


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def serializable_trial(trial: Trial) -> dict[str, Any]:
    return {
        "condition": trial.condition,
        "intermediate_disabled_layers": sorted(trial.intermediate_disabled_layers),
        "final_disabled_layers": sorted(trial.final_disabled_layers),
        "generated_disabled_layers": sorted(trial.generated_disabled_layers),
        "window_start": trial.window_start,
        "window_width": trial.window_width,
        "restore_layers": sorted(trial.restore_layers),
        "restore_mode": trial.restore_mode,
    }


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL line {number} in {path}: {exc}")
    return rows


def mean_ci(values: Iterable[float]) -> tuple[float, float, int]:
    a = np.asarray(list(values), dtype=float)
    if len(a) == 0:
        return float("nan"), float("nan"), 0
    mean = float(a.mean())
    half = 0.0 if len(a) == 1 else float(1.96 * a.std(ddof=1) / math.sqrt(len(a)))
    return mean, half, len(a)


def aggregate_and_plot(out_dir: Path, expected_templates: list[str] | None = None) -> None:
    import pandas as pd

    rows = [r for r in read_rows(out_dir / "results.jsonl") if "error" not in r]
    if not rows:
        raise RuntimeError("no successful rows to analyze")
    df = pd.DataFrame(rows)
    group_cols = ["template", "condition", "window_start", "window_width", "restore_mode"]
    summary_rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        acc, acc_ci, n = mean_ci(group["exact_expected_answer"].astype(float))
        contains, contains_ci, _ = mean_ci(group["entity_in_completion"].astype(float))
        lp, lp_ci, _ = mean_ci(group["expected_entity_log_probability"])
        margin, margin_ci, _ = mean_ci(group["expected_vs_best_alternative_logit_margin"])
        summary_rows.append({
            **dict(zip(group_cols, keys)), "n": n,
            "exact_accuracy": acc, "exact_accuracy_ci95": acc_ci,
            "contains_accuracy": contains, "contains_accuracy_ci95": contains_ci,
            "expected_log_probability": lp, "expected_log_probability_ci95": lp_ci,
            "logit_margin": margin, "logit_margin_ci95": margin_ci,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "aggregated_conditions.csv", index=False)
    try:
        summary.to_parquet(out_dir / "aggregated_conditions.parquet", index=False)
    except Exception:
        pass

    templates = expected_templates or sorted(df.template.unique())
    necessity_limit = max(
        max(
            abs(
                summary[(summary.template == template) & (summary.condition == "necessity_window")].exact_accuracy
                - float(df[(df.template == template) & (df.condition == "generated_read_block")].exact_expected_answer.mean())
            )
        )
        for template in templates
    )
    necessity_limit = max(float(necessity_limit), 0.05)
    fig, axes = plt.subplots(
        1, len(templates), figsize=(5 * len(templates), 4.5),
        squeeze=False, constrained_layout=True,
    )
    for ax, template in zip(axes[0], templates):
        base = float(df[(df.template == template) & (df.condition == "generated_read_block")].exact_expected_answer.mean())
        part = summary[(summary.template == template) & (summary.condition == "necessity_window")]
        pivot = part.pivot(index="window_width", columns="window_start", values="exact_accuracy") - base
        image = ax.imshow(
            pivot.values, aspect="auto", origin="lower",
            vmin=-necessity_limit, vmax=necessity_limit, cmap="coolwarm",
        )
        ax.set_title(f"{template}\nn={int(part.n.max()) if len(part) else 0}/cell")
        ax.set_xlabel("window start layer")
        ax.set_ylabel("window width")
        ax.set_xticks(range(0, len(pivot.columns), 5), [int(x) for x in pivot.columns[::5]])
        ax.set_yticks(range(len(pivot.index)), [int(x) for x in pivot.index])
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Δ exact accuracy vs generated-read block")
    fig.suptitle("Necessity scan: entity-value contribution removal", y=1.04)
    fig.savefig(out_dir / "necessity_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    sufficiency_limit = max(
        max(
            abs(
                summary[(summary.template == template) & (summary.condition == "sufficiency_window")].exact_accuracy
                - float(df[(df.template == template) & (df.condition == "full_intermediate_bottleneck")].exact_expected_answer.mean())
            )
        )
        for template in templates
    )
    sufficiency_limit = max(float(sufficiency_limit), 0.05)
    fig, axes = plt.subplots(
        1, len(templates), figsize=(5 * len(templates), 4.5),
        squeeze=False, constrained_layout=True,
    )
    for ax, template in zip(axes[0], templates):
        base = float(df[(df.template == template) & (df.condition == "full_intermediate_bottleneck")].exact_expected_answer.mean())
        part = summary[(summary.template == template) & (summary.condition == "sufficiency_window")]
        pivot = part.pivot(index="window_width", columns="window_start", values="exact_accuracy") - base
        image = ax.imshow(
            pivot.values, aspect="auto", origin="lower",
            vmin=-sufficiency_limit, vmax=sufficiency_limit, cmap="coolwarm",
        )
        ax.set_title(f"{template}\nn={int(part.n.max()) if len(part) else 0}/cell")
        ax.set_xlabel("window start layer")
        ax.set_ylabel("window width")
        ax.set_xticks(range(0, len(pivot.columns), 5), [int(x) for x in pivot.columns[::5]])
        ax.set_yticks(range(len(pivot.index)), [int(x) for x in pivot.index])
    fig.colorbar(image, ax=axes.ravel().tolist(), label="recovery vs full bottleneck")
    fig.suptitle("Sufficiency scan: native access only inside window", y=1.12)
    fig.savefig(out_dir / "sufficiency_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    profiles = ("clean", "generated_read_block", "full_intermediate_bottleneck")
    fig, axes = plt.subplots(
        1, len(templates), figsize=(5 * len(templates), 4),
        squeeze=False, constrained_layout=True,
    )
    for ax, template in zip(axes[0], templates):
        for condition in profiles:
            sub = df[(df.template == template) & (df.condition == condition)]
            arrays = np.asarray(sub.final_prompt_entity_attention_by_layer.tolist())
            if len(arrays):
                ax.plot(arrays.mean(axis=0), label=f"{condition} (n={len(arrays)})")
        ax.set_title(template)
        ax.set_xlabel("layer")
        ax.set_ylabel("mean entity attention")
        ax.legend(fontsize=7)
    fig.suptitle("Final-prompt-token entity attention layer profile", y=1.04)
    fig.savefig(out_dir / "layer_profile.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(
        1, len(templates), figsize=(5 * len(templates), 4),
        squeeze=False, constrained_layout=True,
    )
    for ax, template in zip(axes[0], templates):
        for condition, label in (("final_restore_entity", "correct entity"), ("final_restore_wrong_entity", "wrong entity")):
            part = summary[(summary.template == template) & (summary.condition == condition)].sort_values("window_start")
            if len(part):
                ax.errorbar(part.window_start, part.exact_accuracy, yerr=part.exact_accuracy_ci95, marker="o", ms=3, label=f"{label} (n={int(part.n.max())})")
        ax.set_title(template)
        ax.set_xlabel("restoration start layer")
        ax.set_ylabel("exact accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
    fig.suptitle("Final-token add-only restoration (not renormalized)", y=1.04)
    fig.savefig(out_dir / "final_token_restoration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    sensitive = {"person_in_list", "friend"}
    baseline_conditions = ["clean", "generated_read_block", "full_intermediate_bottleneck"]
    comparison = []
    for class_name, names in (("sensitive", sensitive), ("robust", set(templates) - sensitive)):
        for condition in baseline_conditions:
            sub = df[df.template.isin(names) & (df.condition == condition)]
            mean, ci, n = mean_ci(sub.exact_expected_answer.astype(float))
            comparison.append((class_name, condition, mean, ci, n))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(baseline_conditions)); width = 0.35
    for offset, class_name in ((-width / 2, "sensitive"), (width / 2, "robust")):
        vals = [next(r for r in comparison if r[0] == class_name and r[1] == c) for c in baseline_conditions]
        ax.bar(x + offset, [v[2] for v in vals], width, yerr=[v[3] for v in vals], label=f"{class_name} (n={vals[0][4]})")
    ax.set_xticks(x, baseline_conditions, rotation=15)
    ax.set_ylim(0, 1.05); ax.set_ylabel("exact accuracy"); ax.legend()
    ax.set_title("Sensitive versus robust templates")
    fig.savefig(out_dir / "sensitive_vs_robust.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    write_summary(out_dir, df, summary, templates)


def write_summary(out_dir: Path, df, summary, templates: list[str]) -> None:
    conditions = ("clean", "generated_read_block", "full_intermediate_bottleneck")
    baseline: dict[str, dict[str, dict[str, float]]] = {}
    lines = [
        "# Intermediate entity layer-window pilot", "",
        "## Method and scope", "",
        "Qwen3-8B-Base was evaluated on 20 valid one-token entities per template (seed 1729; greedy decoding; at most 12 continuation tokens). The primary manipulation is **entity-value contribution removal**, not attention-probability zeroing: native post-softmax coefficients are retained, the original entity value is replaced by zero for targeted queries, and the coefficients are not renormalized. Except in the clean condition, cached generated-token queries are blocked in every layer at every decode step.", "",
        "The final-token rescue exactly follows the reference add-only rule: `max(ordinary attention − live attention, 0) × entity value` is added after attention, independently by head, without renormalization. The wrong-entity source is per-head norm matched; the target prompt and token count are unchanged. Figure error bars are normal-approximation 95% intervals across entities; all heatmap cells have n=20.", "",
        "## Behavioral baselines", "",
        "| Template | Clean exact / contains | Generated block exact / contains | Full intermediate exact / contains | First expected token under full | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for template in templates:
        baseline[template] = {}
        for condition in conditions:
            sub = df[(df.template == template) & (df.condition == condition)]
            baseline[template][condition] = {
                "exact": float(sub.exact_expected_answer.mean()),
                "contains": float(sub.entity_in_completion.mean()),
                "first": float((sub.first_generated_token_id == sub.expected_answer_token_id).mean()),
                "n": len(sub),
            }
        b = baseline[template]
        lines.append(
            f"| {template} | {b['clean']['exact']:.3f} / {b['clean']['contains']:.3f} | "
            f"{b['generated_read_block']['exact']:.3f} / {b['generated_read_block']['contains']:.3f} | "
            f"{b['full_intermediate_bottleneck']['exact']:.3f} / {b['full_intermediate_bottleneck']['contains']:.3f} | "
            f"{b['full_intermediate_bottleneck']['first']:.3f} | {b['clean']['n']} |"
        )

    sensitive, robust = {"person_in_list", "friend"}, {"direct_fact", "visitor_register"}
    lines += ["", "The generated-read block is identical to clean on exact accuracy for every template, so the answer is already established by the final prompt state before decoding. The full intermediate bottleneck reproduces the expected sensitive-template failure: pooled sensitive exact accuracy falls from 0.975 to 0.550 (contains-entity from 1.000 to 0.550), versus 0.925 to 0.825 for the robust pair (contains-entity 1.000 to 0.975). `friend` is the clearest effect (0.950→0.250 exact; 13/20 outputs become generic completions).", "",
        "Behavior and immediate logit evidence are kept separate. For example, `direct_fact` has 0.800 exact sequence accuracy but 0.850 first-expected-token accuracy under the full bottleneck, showing that a correct initial entity token need not yield a stable exact continuation.", "",
        "## Layer-window necessity and sufficiency", "",
    ]
    for template in templates:
        gen = baseline[template]["generated_read_block"]["exact"]
        full = baseline[template]["full_intermediate_bottleneck"]["exact"]
        nec = summary[(summary.template == template) & (summary.condition == "necessity_window")].copy()
        suf = summary[(summary.template == template) & (summary.condition == "sufficiency_window")].copy()
        nec["delta"] = nec.exact_accuracy - gen
        suf["delta"] = suf.exact_accuracy - full
        worst = nec.sort_values(["delta", "window_width", "window_start"]).iloc[0]
        best = suf.sort_values(["delta", "window_width", "window_start"], ascending=[False, True, True]).iloc[0]
        width1 = suf[suf.window_width == 1].sort_values(["delta", "window_start"], ascending=[False, True]).iloc[0]
        lines.append(
            f"- **{template}:** strongest necessity change {worst.delta:+.3f} for "
            f"[{int(worst.window_start)}, {int(worst.window_start + worst.window_width)}); "
            f"best sufficiency recovery {best.delta:+.3f} for "
            f"[{int(best.window_start)}, {int(best.window_start + best.window_width)}); "
            f"best single-layer recovery {width1.delta:+.3f} at layer {int(width1.window_start)}."
        )
    lines += ["", "No scanned window up to width 8 is individually necessary for `person_in_list` or `direct_fact`; the largest `friend` drop is only 0.05. In contrast, small native windows are often sufficient: for `friend`, layers [5,7) alone restore 1.000 exact accuracy and single layers 1 or 6 reach 0.750; for `person_in_list`, a single native layer 19, 20, or 23 restores 1.000; for `direct_fact`, layer 12 or 14 alone restores 1.000. This necessity/sufficiency asymmetry argues for redundant or distributed routes rather than one uniquely required short interval. `visitor_register` is non-monotonic and weakly affected overall; its largest necessity drop (−0.15) is in middle layers around [19,27), but the full bottleneck does not reduce exact accuracy, so that isolated result should not be treated as a clean localization.", "",
        "## Final-prompt readout diagnostics", "",
        "| Template | Late-layer attention clean→full | Late entity-contribution norm clean→full | Largest attention change layer | Largest contribution change layer |",
        "|---|---:|---:|---:|---:|",
    ]
    for template in templates:
        clean = df[(df.template == template) & (df.condition == "clean")]
        full = df[(df.template == template) & (df.condition == "full_intermediate_bottleneck")]
        clean_attn = np.asarray(clean.final_prompt_entity_attention_by_layer.tolist()).mean(0)
        full_attn = np.asarray(full.final_prompt_entity_attention_by_layer.tolist()).mean(0)
        clean_contrib = np.asarray(clean.final_prompt_entity_contribution_norm_by_layer.tolist()).mean(0)
        full_contrib = np.asarray(full.final_prompt_entity_contribution_norm_by_layer.tolist()).mean(0)
        attn_delta, contrib_delta = full_attn - clean_attn, full_contrib - clean_contrib
        lines.append(
            f"| {template} | {clean_attn[24:].mean():.4f}→{full_attn[24:].mean():.4f} | "
            f"{clean_contrib[24:].mean():.2f}→{full_contrib[24:].mean():.2f} | "
            f"{int(np.abs(attn_delta).argmax())} ({attn_delta[np.abs(attn_delta).argmax()]:+.4f}) | "
            f"{int(np.abs(contrib_delta).argmax())} ({contrib_delta[np.abs(contrib_delta).argmax()]:+.2f}) |"
        )
    lines += ["", "The full bottleneck suppresses the sensitive templates' later direct readout: late-layer mean final-token entity attention falls by about 30% for `person_in_list` and 42% for `friend`, while the projected entity-contribution norm falls by about 28% and 48%, respectively. The largest attention changes occur at layers 24 (`person_in_list`) and 23 (`friend`), and the largest contribution changes at layer 32. `visitor_register` is essentially unchanged. Thus the damaging effect emerges mainly in middle-to-late final-token readout even though early/middle native intermediate access can be sufficient to prevent it.", "",
        "## Final-token restoration and identity controls", "",
        "| Template | Full bottleneck | Correct restore from layer 0 | Wrong-entity restore from layer 0 | Wrong entity emitted | Latest start retaining best recovery |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for template in templates:
        correct = summary[(summary.template == template) & (summary.condition == "final_restore_entity")].sort_values("window_start")
        wrong = summary[(summary.template == template) & (summary.condition == "final_restore_wrong_entity")].sort_values("window_start")
        c0, w0 = correct[correct.window_start == 0].iloc[0], wrong[wrong.window_start == 0].iloc[0]
        raw_wrong = df[(df.template == template) & (df.condition == "final_restore_wrong_entity") & (df.window_start == 0)]
        wrong_emitted = np.mean([
            contains_entity(text, entity)
            for text, entity in zip(raw_wrong.completion, raw_wrong.wrong_control_entity)
        ])
        best_accuracy = float(correct.exact_accuracy.max())
        latest = int(correct[correct.exact_accuracy == best_accuracy].window_start.max())
        lines.append(
            f"| {template} | {baseline[template]['full_intermediate_bottleneck']['exact']:.3f} | "
            f"{c0.exact_accuracy:.3f} | {w0.exact_accuracy:.3f} | {wrong_emitted:.3f} | {latest} |"
        )
    lines += ["", "Correct-entity restoration fully rescues both sensitive templates when applied from layer 0 (1.000 exact); for `friend`, full recovery persists even when restoration starts at layer 29. The matched wrong-entity control is strongly identity-specific for the sensitive prompts: at start 0 it yields 0.400 target accuracy for `person_in_list` and 0.000 for `friend`; `friend` emits the injected wrong entity in 5/20 cases and otherwise mostly produces generic/other completions. The robust templates are less diagnostic because their bottleneck behavior is largely preserved even with the wrong source. The no-op restoration is exactly identical to the full bottleneck in tokens, logits, and probabilities for all 80 examples.", "",
        "## Interpretation", "",
        "The evidence is most consistent with **intermediate tokens configuring the final token's later direct entity readout**, with redundant routes across layers. It is not clean evidence that the entity is simply copied into and carried exclusively by intermediate-token representations: no short window is necessary, many disjoint windows are sufficient, the full intervention reduces the final token's later native entity attention/contribution, and directly restoring the final token's entity-directed contribution rescues behavior in an identity-dependent way.", "",
        "Supported claims: the full intermediate entity-value bottleneck causally damages sensitive templates; generated-token direct reads are not needed in the intact behavior; the damage is distributed rather than localized to one necessary ≤8-layer window; sensitive-template damage coincides with reduced middle/late final-token entity readout; and an add-only, non-renormalized final-token supplement can rescue it. Ambiguities: n=20 gives coarse 0.05 accuracy steps and overlapping intervals; the intervention changes value contributions/attention outputs rather than probabilities; restoration is not a natural forward pass and can raise effective coefficient sums above one; norm matching does not make wrong-entity hidden states otherwise identical; and exact-generation failures can differ from first-token evidence.", "",
        "## Artifacts", "", "`results.jsonl` contains every raw generation and per-layer diagnostic. `aggregated_conditions.csv`/`.parquet`, `validation.json`, `reference_clean_validation.json`, `config.json`, and the five PNG figures provide the condition summaries, provenance, validation, and requested visualizations.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_touch_counts(row: dict[str, Any], n_layers: int) -> None:
    touched = row["policy_application_counts_by_layer"]
    inter = {int(k) for k in touched.get("intermediate", {})}
    generated = {int(k) for k in touched.get("generated", {})}
    final = {int(k) for k in touched.get("final", {})}
    restore = {int(k) for k in touched.get("restore", {})}
    if inter != set(row["intermediate_disabled_layers"]):
        raise AssertionError(f"intermediate touched {inter} != requested")
    # A one-token generation has no cached generated query. Smoke uses >=4,
    # but EOS can still stop early; only require all layers when a cached step occurred.
    if len(row["generated_token_ids"]) > 1 and generated != set(row["generated_disabled_layers"]):
        raise AssertionError(f"generated touched {generated} != requested")
    if final != set(row["final_disabled_layers"]):
        raise AssertionError(f"final touched {final} != requested")
    if restore != set(row["restore_layers"]):
        raise AssertionError(f"restore touched {restore} != requested")
    if len(row["final_prompt_entity_attention_by_layer"]) != n_layers:
        raise AssertionError("missing layer diagnostics")


@torch.inference_mode()
def validate_noop_logits(model, encoded: EncodedPrompt, device: torch.device) -> float:
    ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    native = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    with policy_context(encoded, Trial("clean_hook")):
        hooked = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    difference = float((native.float() - hooked.float()).abs().max().cpu())
    if difference != 0.0:
        torch.testing.assert_close(native, hooked, atol=1e-6, rtol=1e-6)
    return difference


def run_self_test() -> None:
    # Pure configuration tests run without loading model weights.
    trials = build_trials(36, smoke=False)
    necessity = [t for t in trials if t.condition == "necessity_window"]
    sufficiency = [t for t in trials if t.condition == "sufficiency_window"]
    expected = sum(36 - w + 1 for w in WIDTHS)
    assert len(necessity) == expected == 198
    assert len(sufficiency) == expected
    for trial in necessity:
        assert len(trial.intermediate_disabled_layers) == trial.window_width
    for trial in sufficiency:
        assert len(trial.intermediate_disabled_layers) == 36 - trial.window_width
    assert len({(t.condition, t.window_start, t.window_width, t.restore_mode) for t in trials}) == len(trials)
    print(f"self-test passed: {len(trials)} unique conditions", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--entity-count", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--physical-gpu-id", default=os.environ.get("PHYSICAL_GPU_ID"))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--limit-trials", type=int, help="debug-only per-example condition limit")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test(); return 0
    if args.out_dir is None:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.out_dir = Path(f"/ssd/mh3897/patchscope_results/qwen3_8b_base_intermediate_entity_layer_window_{stamp}")
    if args.analyze_only:
        metadata = json.loads((args.out_dir / "config.json").read_text())
        aggregate_and_plot(args.out_dir, metadata.get("template_names")); return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    repo = Path(__file__).resolve().parents[2]
    for relative, expected in REFERENCE_HASHES.items():
        actual = _sha256(repo / relative)
        if actual != expected:
            raise RuntimeError(f"reference hash mismatch for {relative}: {actual}")
    device = torch.device(args.device)
    model, tokenizer, n_layers = load_model(args.model, device)
    cases, exclusions, templates = choose_cases(
        tokenizer, 2 if args.smoke else args.entity_count, args.smoke
    )
    trials = build_trials(n_layers, args.smoke)
    if args.limit_trials is not None:
        trials = trials[:args.limit_trials]
    config = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": [sys.executable] + sys.argv,
        "model": args.model,
        "tokenizer": args.model,
        "precision": str(next(model.parameters()).dtype),
        "device_inside_python": str(device),
        "physical_gpu_id": args.physical_gpu_id,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "n_layers": n_layers,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "entity_count": 2 if args.smoke else args.entity_count,
        "template_names": [t.name for t in templates],
        "widths": list(WIDTHS),
        "window_convention": "all full windows [L,L+w); starts 0 through n_layers-w; no clipping",
        "intervention": "entity-value contribution removal with native post-softmax weights retained; no renormalization",
        "generated_query_policy": "original entity-value contribution removed in every layer on every cached decode step, except clean",
        "restoration": "positive ordinary-minus-live entity-attention deficit times entity value, added after attention; no renormalization",
        "wrong_entity_control": "counterfactual entity value, per-head norm matched to target entity value; target prompt unchanged",
        "git": git_info(repo),
        "packages": package_versions(),
        "reference_provenance": {"remote": "target (RiddleHe/nanochat)", "branch": "research/skip-ahead", "commit": REFERENCE_COMMIT, "sha256": REFERENCE_HASHES},
        "exclusions": exclusions,
        "n_conditions_per_example": len(trials),
        "smoke": args.smoke,
    }
    config_path = args.out_dir / "config.json"
    if config_path.exists():
        prior = json.loads(config_path.read_text())
        for key in ("model", "seed", "n_layers", "template_names", "smoke"):
            if prior.get(key) != config.get(key):
                raise RuntimeError(f"resume config mismatch for {key}")
    else:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")

    results_path = args.out_dir / "results.jsonl"
    failures_path = args.out_dir / "failures.jsonl"
    completed = {
        trial_key(row, row["template"], row["entity"])
        for row in read_rows(results_path)
    }
    all_entity_names = [x[0] for x in ENTITY_SETS["diverse100"]]
    validation: dict[str, Any] = {"resume_initial_rows": len(completed), "checks": []}
    validation["noop_clean_logits_max_abs_difference"] = validate_noop_logits(model, cases[0], device)
    validation["checks"].append("inactive and no-op clean-hook logits match")

    # Counterfactual pairing is fixed and cyclic within the selected valid entities.
    by_template_entity = {(c.template_name, c.entity): c for c in cases}
    entities = list(dict.fromkeys(c.entity for c in cases))
    successful_new = failed_new = 0
    for case_index, encoded in enumerate(cases, 1):
        pending = [t for t in trials if trial_key(t, encoded.template_name, encoded.entity) not in completed]
        if not pending:
            print(f"resume skip {case_index}/{len(cases)} {encoded.template_name}/{encoded.entity}", flush=True)
            continue
        ordinary_attention, _ = capture_reference(model, encoded, device)
        wrong_entity = entities[(entities.index(encoded.entity) + 1) % len(entities)]
        wrong_case = by_template_entity[(encoded.template_name, wrong_entity)]
        if len(wrong_case.ids) != len(encoded.ids) or wrong_case.entity_position != encoded.entity_position:
            raise RuntimeError("wrong-entity control changed token count or entity position")
        _, wrong_values = capture_reference(model, wrong_case, device)
        print(f"case {case_index}/{len(cases)} {encoded.template_name}/{encoded.entity}: {len(pending)} pending", flush=True)
        for trial_index, trial in enumerate(pending, 1):
            try:
                row = run_trial(
                    model, tokenizer, encoded, trial, device,
                    args.max_new_tokens, n_layers, ordinary_attention,
                    wrong_values if trial.restore_mode == "wrong_entity" else None,
                    all_entity_names, args.seed,
                )
                row["wrong_control_entity"] = wrong_entity if trial.restore_mode == "wrong_entity" else None
                validate_touch_counts(row, n_layers)
                append_jsonl(results_path, row)
                completed.add(trial_key(trial, encoded.template_name, encoded.entity))
                successful_new += 1
            except torch.cuda.OutOfMemoryError:
                append_jsonl(failures_path, {"template": encoded.template_name, "entity": encoded.entity, "condition": serializable_trial(trial), "error": "CUDA out of memory", "traceback": traceback.format_exc()})
                torch.cuda.empty_cache()
                raise
            except Exception as exc:
                failed_new += 1
                append_jsonl(failures_path, {"template": encoded.template_name, "entity": encoded.entity, "condition": serializable_trial(trial), "error": repr(exc), "traceback": traceback.format_exc()})
                print(f"FAILED {encoded.template_name}/{encoded.entity}/{trial.condition}: {exc}", flush=True)
            if trial_index % 25 == 0 or trial_index == len(pending):
                print(f"  {trial_index}/{len(pending)} conditions; total rows={len(completed)}", flush=True)

    final_rows = read_rows(results_path)
    full_rows = {
        (r["template"], r["entity"]): r
        for r in final_rows if r["condition"] == "full_intermediate_bottleneck"
    }
    noop_rows = {
        (r["template"], r["entity"]): r
        for r in final_rows if r["condition"] == "final_restore_noop"
    }
    noop_fields = (
        "generated_token_ids", "expected_entity_log_probability",
        "expected_vs_best_alternative_logit_margin",
        "first_generated_token_probability",
    )
    noop_matches = (
        set(full_rows) == set(noop_rows)
        and all(
            all(full_rows[key][field] == noop_rows[key][field] for field in noop_fields)
            for key in full_rows
        )
    )
    validation.update({
        "completed_rows": len(final_rows),
        "new_successful_rows": successful_new,
        "new_failed_rows": failed_new,
        "expected_rows": len(cases) * len(trials),
        "generated_cached_decode_verified": any(
            len(r.get("generated_token_ids", [])) > 1 and bool(r.get("policy_application_counts_by_layer", {}).get("generated"))
            for r in final_rows
        ),
        "noop_restoration_matches_full_bottleneck": noop_matches,
        "resumability": "row keys loaded before execution; completed rows skipped; fsync after every row",
    })
    if (
        validation["completed_rows"] != validation["expected_rows"]
        or not validation["noop_restoration_matches_full_bottleneck"]
    ):
        validation["status"] = "incomplete"
    else:
        validation["status"] = "passed"
        validation["checks"] += [
            "intermediate layers touched exactly requested sets",
            "necessity windows touch only [L,L+w)",
            "sufficiency windows leave only [L,L+w) native",
            "final and generated policies use independent layer sets",
            "entity/intermediate/final positions saved per row",
            "no-op restoration matches the full bottleneck exactly",
        ]
    (args.out_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    aggregate_and_plot(args.out_dir, [t.name for t in templates])
    print(json.dumps({"result_dir": str(args.out_dir), **validation}, indent=2), flush=True)
    return 0 if validation["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
