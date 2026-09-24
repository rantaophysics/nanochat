"""Fixed confirmatory panel for intermediate-token entity-value sufficiency.

The attention intervention retains native QK scores and post-softmax weights,
zeros every value in the entity-name span, recomputes the attention output, and
substitutes that output only at targeted query positions.  It never zeros or
renormalizes attention probabilities.  Raw rows are fsynced individually and
the run is safely resumable.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import re
import socket
import subprocess
import sys
import time
import traceback
import unicodedata
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


MODEL = "Qwen/Qwen3-8B-Base"
SEED = 1729
N_LAYERS = 36
MAX_NEW_TOKENS = 12
TEMPLATE_NAMES = (
    "direct_fact", "person_in_list", "friend", "visitor_register", "dialogue"
)
CONDITION_NAMES = (
    "clean",
    "generated_read_block",
    "full_intermediate_bottleneck",
    "early_5_7",
    "middle_12_14",
    "later_middle_19_21",
    "sparse_union",
    "late_negative_30_32",
)
POSITIVE_CONDITIONS = (
    "early_5_7", "middle_12_14", "later_middle_19_21", "sparse_union"
)
CANDIDATE_CONDITIONS = POSITIVE_CONDITIONS + ("late_negative_30_32",)
REFERENCE_ROOT = Path(
    "/hdd/mh3897/nanochat-skip-ahead/results/relay-qwen3-8b/"
    "11-intermediate-blocked-readout-layer-span-enable-output/source"
)
DEFAULT_ENTITIES = REFERENCE_ROOT / "entities100.json"
DEFAULT_REFERENCE = REFERENCE_ROOT / "qwen_entity_attention_ablation.py"
PILOT_BASENAME = "qwen3_8b_base_intermediate_entity_layer_window_20260923_043856"
ATTENTION_IMPLEMENTATION = "intermediate_window_confirmation"
ALL_LAYERS = frozenset(range(N_LAYERS))
SPARSE_UNION = frozenset({5, 6, 12, 13, 19, 20})


@dataclass(frozen=True)
class Trial:
    condition: str
    native_intermediate_layers: frozenset[int]
    intermediate_disabled_layers: frozenset[int]
    generated_disabled_layers: frozenset[int]
    final_disabled_layers: frozenset[int] = frozenset()


@dataclass
class PolicyState:
    active: bool = False
    entity_positions: tuple[int, ...] = ()
    intermediate_positions: tuple[int, ...] = ()
    final_position: int | None = None
    prompt_length: int | None = None
    intermediate_disabled_layers: frozenset[int] = frozenset()
    generated_disabled_layers: frozenset[int] = frozenset()
    final_disabled_layers: frozenset[int] = frozenset()
    diagnostics: dict[int, dict[str, float]] = field(default_factory=dict)
    touched: dict[str, dict[int, int]] = field(default_factory=dict)


POLICY = PolicyState()


def fixed_trials(n_layers: int = N_LAYERS) -> list[Trial]:
    layers = frozenset(range(n_layers))

    def trial(name: str, native: frozenset[int], block_generated: bool) -> Trial:
        return Trial(
            name,
            native,
            layers - native,
            layers if block_generated else frozenset(),
            frozenset(),
        )

    return [
        trial("clean", layers, False),
        trial("generated_read_block", layers, True),
        trial("full_intermediate_bottleneck", frozenset(), True),
        trial("early_5_7", frozenset({5, 6}), True),
        trial("middle_12_14", frozenset({12, 13}), True),
        trial("later_middle_19_21", frozenset({19, 20}), True),
        trial("sparse_union", frozenset({5, 6, 12, 13, 19, 20}), True),
        trial("late_negative_30_32", frozenset({30, 31}), True),
    ]


def load_reference(path: Path):
    spec = importlib.util.spec_from_file_location("confirmation_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import reference source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_entities(path: Path) -> list[tuple[str, str]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != 100:
        raise ValueError(f"entity manifest must contain exactly 100 rows, got {len(records) if isinstance(records, list) else 'non-list'}")
    entities: list[tuple[str, str]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"entity row {index} is not an object")
        entity, group = record.get("entity"), record.get("entity_group")
        if not isinstance(entity, str) or not entity or entity != entity.strip():
            raise ValueError(f"invalid entity at row {index}: {entity!r}")
        if not isinstance(group, str) or not group:
            raise ValueError(f"invalid entity group at row {index}")
        entities.append((entity, group))
    names = [name for name, _ in entities]
    if len(set(names)) != 100:
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"entity manifest is not unique: {duplicates}")
    return entities


def selected_templates(reference) -> list[Any]:
    by_name = {template.name: template for template in reference.TEMPLATES}
    missing = set(TEMPLATE_NAMES) - set(by_name)
    if missing:
        raise RuntimeError(f"reference source lacks templates: {sorted(missing)}")
    return [by_name[name] for name in TEMPLATE_NAMES]


def run_self_test(entities_path: Path, reference_path: Path) -> dict[str, Any]:
    reference = load_reference(reference_path)
    entities = load_entities(entities_path)
    templates = selected_templates(reference)
    trials = fixed_trials()
    assert [t.name for t in templates] == list(TEMPLATE_NAMES)
    assert len(trials) == 8
    assert [t.condition for t in trials] == list(CONDITION_NAMES)
    assert len(set(t.condition for t in trials)) == 8
    expected_native = {
        "clean": ALL_LAYERS,
        "generated_read_block": ALL_LAYERS,
        "full_intermediate_bottleneck": frozenset(),
        "early_5_7": frozenset({5, 6}),
        "middle_12_14": frozenset({12, 13}),
        "later_middle_19_21": frozenset({19, 20}),
        "sparse_union": SPARSE_UNION,
        "late_negative_30_32": frozenset({30, 31}),
    }
    for trial in trials:
        assert trial.native_intermediate_layers == expected_native[trial.condition]
        assert trial.intermediate_disabled_layers == ALL_LAYERS - expected_native[trial.condition]
        assert not trial.final_disabled_layers
        assert trial.generated_disabled_layers == (
            frozenset() if trial.condition == "clean" else ALL_LAYERS
        )
    assert SPARSE_UNION == frozenset({5, 6, 12, 13, 19, 20})
    assert len(entities) * len(templates) * len(trials) == 4000
    result = {
        "status": "passed",
        "templates": len(templates),
        "conditions": len(trials),
        "unique_entities": len(entities),
        "expected_full_rows": 4000,
        "sparse_union": sorted(SPARSE_UNION),
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, kv_heads, seq_len, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None, :, :].expand(
        batch, kv_heads, n_rep, seq_len, head_dim
    ).reshape(batch, kv_heads * n_rep, seq_len, head_dim)


def _entity_attention(
    module, query: torch.Tensor, key: torch.Tensor,
    attention_mask: torch.Tensor | None, scaling: float,
    local_queries: list[int], entity_positions: tuple[int, ...],
) -> torch.Tensor:
    keys = _repeat_kv(key, module.num_key_value_groups)
    scores = torch.matmul(
        query[:, :, local_queries, :], keys.transpose(2, 3)
    ) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, local_queries, : keys.shape[-2]]
    weights = F.softmax(scores, dim=-1, dtype=torch.float32)
    return weights[..., list(entity_positions)]


def _projected_norm(module, head_values: torch.Tensor) -> torch.Tensor:
    flat = head_values.permute(0, 2, 1, 3).reshape(
        head_values.shape[0], head_values.shape[2], -1
    )
    projected = F.linear(flat, module.o_proj.weight, bias=None)
    return torch.linalg.vector_norm(projected.float(), dim=-1)


def _bump(kind: str, layer: int, count: int) -> None:
    if count:
        target = POLICY.touched.setdefault(kind, {})
        target[layer] = target.get(layer, 0) + count


def confirmation_attention_forward(
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
    if not POLICY.entity_positions or POLICY.final_position is None:
        raise RuntimeError("active policy has incomplete prompt positions")
    layer = int(module.layer_idx)
    q_len, k_len = query.shape[-2], key.shape[-2]
    q_start = k_len - q_len
    absolute = list(range(q_start, q_start + q_len))
    final_local = [i for i, pos in enumerate(absolute) if pos == POLICY.final_position]
    intermediate_local = [
        i for i, pos in enumerate(absolute) if pos in POLICY.intermediate_positions
    ]
    generated_local = [
        i for i, pos in enumerate(absolute) if pos >= (POLICY.prompt_length or 0)
    ]
    values = _repeat_kv(value, module.num_key_value_groups)

    if final_local:
        if len(final_local) != 1 or layer in POLICY.diagnostics:
            raise RuntimeError(f"invalid prompt diagnostic at layer {layer}")
        final_attention_by_token = _entity_attention(
            module, query, key, attention_mask, scaling,
            final_local, POLICY.entity_positions,
        )
        final_weighted = (
            final_attention_by_token.to(values.dtype).unsqueeze(-1)
            * values[:, :, list(POLICY.entity_positions), :].unsqueeze(2)
        ).sum(dim=3)
        if intermediate_local:
            inter_attention = _entity_attention(
                module, query, key, attention_mask, scaling,
                intermediate_local, POLICY.entity_positions,
            )
            inter_weighted = (
                inter_attention.to(values.dtype).unsqueeze(-1)
                * values[:, :, list(POLICY.entity_positions), :].unsqueeze(2)
            ).sum(dim=3)
            inter_norm = _projected_norm(module, inter_weighted)[0]
            inter_mean, inter_sum = float(inter_norm.mean().cpu()), float(inter_norm.sum().cpu())
        else:
            inter_mean = inter_sum = 0.0
        summed_attention = final_attention_by_token.sum(dim=-1)
        POLICY.diagnostics[layer] = {
            "final_entity_attention_mean": float(summed_attention.mean().cpu()),
            "final_entity_attention_max": float(summed_attention.max().cpu()),
            "final_entity_contribution_norm": float(
                _projected_norm(module, final_weighted)[0, 0].cpu()
            ),
            "intermediate_entity_contribution_mean_norm": inter_mean,
            "intermediate_entity_contribution_sum_norm": inter_sum,
        }

    affected_intermediate = (
        intermediate_local if layer in POLICY.intermediate_disabled_layers else []
    )
    affected_generated = (
        generated_local if layer in POLICY.generated_disabled_layers else []
    )
    affected_final = final_local if layer in POLICY.final_disabled_layers else []
    _bump("intermediate", layer, len(affected_intermediate))
    _bump("generated", layer, len(affected_generated))
    _bump("final", layer, len(affected_final))
    affected = sorted(set(affected_intermediate + affected_generated + affected_final))
    if not affected:
        return sdpa_attention_forward(
            module, query, key, value, attention_mask,
            scaling=scaling, dropout=dropout, **kwargs,
        )
    zero_value = value.clone()
    zero_value[:, :, list(POLICY.entity_positions), :] = 0
    zero_output, _ = sdpa_attention_forward(
        module, query, key, zero_value, attention_mask,
        scaling=scaling, dropout=dropout, **kwargs,
    )
    if len(affected) == q_len:
        return zero_output, None
    native_output, _ = sdpa_attention_forward(
        module, query, key, value, attention_mask,
        scaling=scaling, dropout=dropout, **kwargs,
    )
    output = native_output.clone()
    output[:, affected, :, :] = zero_output[:, affected, :, :]
    return output, None


@contextmanager
def policy_context(encoded, trial: Trial):
    if POLICY.active:
        raise RuntimeError("nested policies are unsupported")
    final = len(encoded.ids) - 1
    entity_positions = tuple(encoded.entity_positions)
    intermediate = tuple(range(entity_positions[-1] + 1, final))
    if not intermediate:
        raise ValueError("prompt has no post-entity intermediate positions")
    POLICY.active = True
    POLICY.entity_positions = entity_positions
    POLICY.intermediate_positions = intermediate
    POLICY.final_position = final
    POLICY.prompt_length = len(encoded.ids)
    POLICY.intermediate_disabled_layers = trial.intermediate_disabled_layers
    POLICY.generated_disabled_layers = trial.generated_disabled_layers
    POLICY.final_disabled_layers = trial.final_disabled_layers
    POLICY.diagnostics = {}
    POLICY.touched = {}
    try:
        yield POLICY
    finally:
        POLICY.active = False
        POLICY.entity_positions = ()
        POLICY.intermediate_positions = ()
        POLICY.final_position = None
        POLICY.prompt_length = None
        POLICY.intermediate_disabled_layers = frozenset()
        POLICY.generated_disabled_layers = frozenset()
        POLICY.final_disabled_layers = frozenset()


def load_model(model_name: str, device: torch.device):
    from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

    AttentionInterface.register(ATTENTION_IMPLEMENTATION, confirmation_attention_forward)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, attn_implementation=ATTENTION_IMPLEMENTATION
    ).to(device).eval()
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError("cannot locate transformer layers")
    return model, tokenizer, len(model.model.layers)


def normalize_text(text: str) -> str:
    text = re.sub(r"<\|[^|]+\|>", "", text)
    return unicodedata.normalize("NFKC", text).casefold().strip()


def contains_entity(text: str, entity: str) -> bool:
    return re.search(
        rf"(?<!\w){re.escape(normalize_text(entity))}(?!\w)", normalize_text(text)
    ) is not None


def exact_answer(text: str, entity: str) -> bool:
    visible = normalize_text(text).strip(" \t\r\n.,;:!?\"'`-_")
    return visible == normalize_text(entity)


def partial_name(text: str, entity: str) -> bool:
    parts = [p for p in re.findall(r"\w+", normalize_text(entity)) if len(p) > 1]
    return len(parts) > 1 and any(
        re.search(rf"(?<!\w){re.escape(part)}(?!\w)", normalize_text(text))
        for part in parts
    )


def failure_category(text: str, entity: str, all_entities: list[str]) -> str:
    if exact_answer(text, entity):
        return "correct_exact"
    if contains_entity(text, entity):
        return "correct_identity_strict_form_change"
    norm = normalize_text(text)
    if not norm or not re.search(r"\w", norm):
        return "empty_or_punctuation"
    refusal_terms = (
        "not provided", "not specified", "unknown", "cannot determine",
        "can't determine", "no name", "person's name", "the name",
    )
    if any(term in norm for term in refusal_terms):
        return "generic_refusal_or_not_provided"
    if partial_name(text, entity):
        return "partial_name"
    for other in sorted((x for x in all_entities if x != entity), key=len, reverse=True):
        if contains_entity(text, other):
            return "wrong_distractor_entity"
    if "�" in text:
        return "malformed_text"
    return "other_output_failure"


def expected_target(tokenizer, encoded, entity: str) -> tuple[int, list[int]]:
    target_ids = list(tokenizer(" " + entity, add_special_tokens=False)["input_ids"])
    if not target_ids:
        raise ValueError(f"{entity!r} has no answer tokens")
    prompt_ids = list(tokenizer(encoded.prompt, add_special_tokens=True)["input_ids"])
    joined = list(tokenizer(encoded.prompt + " " + entity, add_special_tokens=True)["input_ids"])
    if joined[: len(prompt_ids)] != prompt_ids or not joined[len(prompt_ids):]:
        raise ValueError(f"{encoded.template_name}/{entity}: answer is not a clean token continuation")
    contextual = joined[len(prompt_ids):]
    if contextual != target_ids:
        raise ValueError(
            f"{encoded.template_name}/{entity}: contextual target {contextual} != standalone-space target {target_ids}"
        )
    # The prespecified next-token target is exactly the first token, even when
    # the complete entity answer contains more than one token.
    return int(target_ids[0]), target_ids


def encode_cases(reference, tokenizer, entities, templates, smoke: bool):
    chosen = [entities[0], entities[50]] if smoke else entities
    chosen_ids = [0, 50] if smoke else list(range(100))
    cases = []
    errors = []
    target_audit = []
    for template in templates:
        for entity_id, (entity, group) in zip(chosen_ids, chosen):
            try:
                encoded = reference.encode_prompt(
                    tokenizer, template, entity_id, entity, group
                )
                expected_id, target_ids = expected_target(tokenizer, encoded, entity)
                if not encoded.entity_positions:
                    raise ValueError("empty entity span")
                target_audit.append({
                    "template": template.name,
                    "entity_id": entity_id,
                    "entity": entity,
                    "entity_positions": list(encoded.entity_positions),
                    "entity_token_ids": [encoded.ids[p] for p in encoded.entity_positions],
                    "expected_target_token_id": expected_id,
                    "complete_answer_token_ids": target_ids,
                    "expected_target_token_count": 1,
                })
                cases.append((encoded, expected_id, target_ids))
            except Exception as exc:
                errors.append({"template": template.name, "entity": entity, "error": repr(exc)})
    if errors:
        raise RuntimeError(
            "entity validation failed; no entities may be excluded:\n" + json.dumps(errors, indent=2)
        )
    expected = 10 if smoke else 500
    if len(cases) != expected or len({c[0].entity for c in cases}) != (2 if smoke else 100):
        raise RuntimeError(f"case construction produced {len(cases)} cases, expected {expected}")
    return cases, target_audit


def _counts(state: PolicyState) -> dict[str, dict[str, int]]:
    return {
        kind: {str(layer): count for layer, count in sorted(values.items())}
        for kind, values in sorted(state.touched.items())
    }


def _generate(model, tokenizer, ids: torch.Tensor, max_new_tokens: int):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    return model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=pad_id,
    )


@torch.inference_mode()
def run_trial(
    model, tokenizer, encoded, expected_id: int, answer_ids: list[int], trial: Trial,
    device: torch.device, max_new_tokens: int, n_layers: int,
    all_entities: list[str], seed: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    with policy_context(encoded, trial) as state:
        generated = _generate(model, tokenizer, ids, max_new_tokens)
        generated_ids = generated.sequences[0, len(encoded.ids):].detach().cpu().tolist()
        if not generated_ids or not generated.scores:
            raise RuntimeError("generation produced no token")
        first_logits = generated.scores[0][0].float()
        log_probs = F.log_softmax(first_logits, dim=-1)
        expected_log_prob = float(log_probs[expected_id].cpu())
        expected_prob = float(log_probs[expected_id].exp().cpu())
        masked = first_logits.clone()
        masked[expected_id] = -torch.inf
        margin = float((first_logits[expected_id] - masked.max()).cpu())
        diagnostics = [state.diagnostics[layer] for layer in range(n_layers)]
        touched = _counts(state)
    completion = tokenizer.decode(generated_ids, skip_special_tokens=False)
    first_id = int(generated_ids[0])
    return {
        "seed": seed,
        "model": MODEL,
        "condition": trial.condition,
        "native_intermediate_layers": sorted(trial.native_intermediate_layers),
        "intermediate_disabled_layers": sorted(trial.intermediate_disabled_layers),
        "generated_disabled_layers": sorted(trial.generated_disabled_layers),
        "final_disabled_layers": sorted(trial.final_disabled_layers),
        "template_id": encoded.template_id,
        "template": encoded.template_name,
        "entity_id": encoded.entity_id,
        "entity": encoded.entity,
        "entity_group": encoded.entity_group,
        "prompt": encoded.prompt,
        "prompt_token_ids": encoded.ids,
        "prompt_tokens": encoded.tokens,
        "entity_token_position": encoded.entity_position,
        "entity_positions": list(encoded.entity_positions),
        "entity_token_ids": [encoded.ids[p] for p in encoded.entity_positions],
        "entity_tokens": [encoded.tokens[p] for p in encoded.entity_positions],
        "intermediate_token_positions": list(range(encoded.entity_positions[-1] + 1, len(encoded.ids) - 1)),
        "final_prompt_token_position": len(encoded.ids) - 1,
        "raw_completion": completion,
        "completion": completion,
        "normalized_completion": normalize_text(completion),
        "generated_token_ids": generated_ids,
        "first_generated_token_id": first_id,
        "first_generated_token": tokenizer.decode([first_id], skip_special_tokens=False),
        "expected_entity_token_id": expected_id,
        "expected_complete_answer_token_ids": answer_ids,
        "exact_expected_answer_correct": exact_answer(completion, encoded.entity),
        "target_entity_present": contains_entity(completion, encoded.entity),
        "first_generated_token_correct": first_id == expected_id,
        "expected_entity_probability": expected_prob,
        "expected_entity_log_probability": expected_log_prob,
        "expected_vs_best_alternative_logit_margin": margin,
        "failure_category": failure_category(completion, encoded.entity, all_entities),
        "final_token_entity_attention_by_layer": [d["final_entity_attention_mean"] for d in diagnostics],
        "final_token_entity_contribution_norm_by_layer": [d["final_entity_contribution_norm"] for d in diagnostics],
        "intermediate_entity_contribution_mean_norm_by_layer": [d["intermediate_entity_contribution_mean_norm"] for d in diagnostics],
        "intermediate_entity_contribution_sum_norm_by_layer": [d["intermediate_entity_contribution_sum_norm"] for d in diagnostics],
        "policy_application_counts_by_query_type_and_layer": touched,
        "runtime_seconds": time.perf_counter() - started,
    }


def trial_key(row_or_trial: dict[str, Any] | Trial, template: str, entity_id: int) -> str:
    condition = row_or_trial.condition if isinstance(row_or_trial, Trial) else row_or_trial.get("condition")
    return json.dumps([int(entity_id), template, condition], separators=(",", ":"))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_rows(path: Path, reject_duplicates: bool = True) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows, seen = [], set()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL {path}:{number}: {exc}") from exc
            if reject_duplicates and {"entity_id", "template", "condition"} <= set(row):
                key = trial_key(row, row["template"], row["entity_id"])
                if key in seen:
                    raise RuntimeError(f"duplicate condition key at {path}:{number}: {key}")
                seen.add(key)
            rows.append(row)
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def package_versions() -> dict[str, str]:
    result = {"python": platform.python_version(), "torch": torch.__version__, "cuda": str(torch.version.cuda)}
    for package in ("transformers", "numpy", "scipy", "pandas", "pyarrow", "matplotlib"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def git_info(repo: Path) -> dict[str, str]:
    def call(*args: str) -> str:
        return subprocess.run(args, cwd=repo, text=True, capture_output=True, check=False).stdout.strip()
    return {
        "commit": call("git", "rev-parse", "HEAD"),
        "branch": call("git", "branch", "--show-current"),
        "status_porcelain": call("git", "status", "--porcelain=v1"),
    }


def copy_provenance(out_dir: Path, entities_path: Path, reference_path: Path) -> dict[str, str]:
    import shutil

    source = out_dir / "source"
    source.mkdir(parents=True, exist_ok=True)
    targets = {
        "entities100.json": entities_path,
        "qwen_entity_attention_ablation.py": reference_path,
        "qwen_intermediate_window_confirmation.py": Path(__file__).resolve(),
    }
    hashes = {}
    for name, origin in targets.items():
        destination = source / name
        if destination.exists() and sha256(destination) != sha256(origin):
            raise RuntimeError(f"provenance destination differs on resume: {destination}")
        if not destination.exists():
            shutil.copy2(origin, destination)
        hashes[f"source/{name}"] = sha256(destination)
    atomic_json(out_dir / "source_hashes.json", hashes)
    return hashes


def find_pilot_results() -> Path | None:
    candidates = [
        Path("/ssd/mh3897/nanochat/results") / PILOT_BASENAME / "results.jsonl",
        Path("/ssd/mh3897/patchscope_results") / PILOT_BASENAME / "results.jsonl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    for root in (Path("/ssd/mh3897/nanochat/results"), Path("/ssd/mh3897/patchscope_results")):
        if root.exists():
            found = list(root.glob(f"**/{PILOT_BASENAME}/results.jsonl"))
            if found:
                return found[0]
    return None


def write_pilot_overlap(out_dir: Path, entities: list[tuple[str, str]]) -> dict[str, Any]:
    path = find_pilot_results()
    current = {entity for entity, _ in entities}
    if path is None:
        report = {"status": "unavailable", "pilot_results": None, "overlap_entities": [], "overlap_count": 0, "confirmatory_sample_size": 100}
    else:
        # The pilot intentionally has many window rows for each
        # entity/template/condition family, so its historical keys are not the
        # fixed-panel keys used by this runner.
        pilot_entities = sorted({row["entity"] for row in read_rows(path, reject_duplicates=False)})
        overlap = sorted(current & set(pilot_entities))
        report = {
            "status": "verified",
            "pilot_results": str(path),
            "pilot_results_sha256": sha256(path),
            "pilot_entities": pilot_entities,
            "pilot_entity_count": len(pilot_entities),
            "overlap_entities": overlap,
            "overlap_count": len(overlap),
            "confirmatory_sample_size": 100 - len(overlap),
            "matching_rule": "exact entity string",
        }
    atomic_json(out_dir / "pilot_overlap.json", report)
    return report


def validate_touch_counts(row: dict[str, Any], n_layers: int) -> None:
    touched = row["policy_application_counts_by_query_type_and_layer"]
    expected_inter = set(row["intermediate_disabled_layers"])
    expected_gen = set(row["generated_disabled_layers"])
    inter = {int(k) for k in touched.get("intermediate", {})}
    generated = {int(k) for k in touched.get("generated", {})}
    final = {int(k) for k in touched.get("final", {})}
    if inter != expected_inter:
        raise AssertionError(f"intermediate touched {sorted(inter)} != {sorted(expected_inter)}")
    intermediate_count = len(row["intermediate_token_positions"])
    if any(v != intermediate_count for v in touched.get("intermediate", {}).values()):
        raise AssertionError("intermediate application count is not one per targeted query")
    cached_steps = max(len(row["generated_token_ids"]) - 1, 0)
    if generated != (expected_gen if cached_steps else set()):
        raise AssertionError(f"generated touched {sorted(generated)} != requested cached policy")
    if any(v != cached_steps for v in touched.get("generated", {}).values()):
        raise AssertionError("generated policy was not applied on every cached step")
    if final or row["final_disabled_layers"]:
        raise AssertionError("final prompt query was modified")
    if len(row["final_token_entity_attention_by_layer"]) != n_layers:
        raise AssertionError("missing per-layer diagnostics")


@torch.inference_mode()
def validate_hook_equivalence(model, encoded, device: torch.device) -> float:
    ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    native = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    with policy_context(encoded, fixed_trials()[0]):
        hooked = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    difference = float((native.float() - hooked.float()).abs().max().cpu())
    if difference != 0.0:
        torch.testing.assert_close(native, hooked, atol=1e-6, rtol=1e-6)
    return difference


def validate_reference_clean(reference, model, tokenizer, encoded, device, max_new_tokens, row) -> None:
    clean = reference.greedy_completion(model, tokenizer, encoded, device, max_new_tokens)
    if clean["generated_token_ids"] != row["generated_token_ids"]:
        raise AssertionError(
            f"clean generation differs from reference for {encoded.template_name}/{encoded.entity}"
        )


def mean_ci(values: Iterable[float]) -> tuple[float, float, float, int]:
    from scipy.stats import t

    array = np.asarray(list(values), dtype=float)
    if not len(array):
        return math.nan, math.nan, math.nan, 0
    mean = float(array.mean())
    if len(array) == 1:
        return mean, mean, mean, 1
    half = float(t.ppf(0.975, len(array) - 1) * array.std(ddof=1) / math.sqrt(len(array)))
    return mean, mean - half, mean + half, len(array)


def wilson_ci(successes: int, n: int) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half, center + half


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [math.nan] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def _primary(template: str, candidate: str) -> bool:
    return candidate == "sparse_union" or (
        (template, candidate) in {
            ("friend", "early_5_7"),
            ("direct_fact", "middle_12_14"),
            ("person_in_list", "later_middle_19_21"),
        }
    )


def paired_row(template: str, candidate: str, baseline: str, left, right, analysis_set: str, kind: str) -> dict[str, Any]:
    from scipy.stats import binomtest

    merged = left.merge(right, on=["entity_id", "template"], suffixes=("_baseline", "_candidate"), validate="one_to_one")
    base_sem = merged.target_entity_present_baseline.astype(bool).to_numpy()
    cand_sem = merged.target_entity_present_candidate.astype(bool).to_numpy()
    base_exact = merged.exact_expected_answer_correct_baseline.astype(bool).to_numpy()
    cand_exact = merged.exact_expected_answer_correct_candidate.astype(bool).to_numpy()

    def transitions(base, cand):
        rescued = int((~base & cand).sum())
        harmed = int((base & ~cand).sum())
        unchanged_correct = int((base & cand).sum())
        unchanged_incorrect = int((~base & ~cand).sum())
        failures = int((~base).sum())
        fraction = rescued / failures if failures else math.nan
        low, high = wilson_ci(rescued, failures)
        p = float(binomtest(min(rescued, harmed), rescued + harmed, 0.5).pvalue) if rescued + harmed else 1.0
        return rescued, harmed, unchanged_correct, unchanged_incorrect, failures, fraction, low, high, p

    sem = transitions(base_sem, cand_sem)
    exact = transitions(base_exact, cand_exact)
    lp_delta = merged.expected_entity_log_probability_candidate - merged.expected_entity_log_probability_baseline
    margin_delta = merged.expected_vs_best_alternative_logit_margin_candidate - merged.expected_vs_best_alternative_logit_margin_baseline
    lp = mean_ci(lp_delta)
    margin = mean_ci(margin_delta)
    return {
        "analysis_set": analysis_set,
        "template": template,
        "contrast_kind": kind,
        "baseline_condition": baseline,
        "candidate_condition": candidate,
        "n": len(merged),
        "semantic_rescued": sem[0], "semantic_harmed": sem[1],
        "semantic_unchanged_correct": sem[2], "semantic_unchanged_incorrect": sem[3],
        "baseline_semantic_failures": sem[4], "conditional_semantic_rescue_fraction": sem[5],
        "conditional_semantic_rescue_ci95_low": sem[6], "conditional_semantic_rescue_ci95_high": sem[7],
        "semantic_mcnemar_exact_p": sem[8],
        "exact_rescued": exact[0], "exact_harmed": exact[1],
        "exact_unchanged_correct": exact[2], "exact_unchanged_incorrect": exact[3],
        "baseline_exact_failures": exact[4], "conditional_exact_rescue_fraction": exact[5],
        "conditional_exact_rescue_ci95_low": exact[6], "conditional_exact_rescue_ci95_high": exact[7],
        "exact_mcnemar_exact_p": exact[8],
        "mean_paired_expected_log_probability_change": lp[0],
        "paired_expected_log_probability_change_ci95_low": lp[1],
        "paired_expected_log_probability_change_ci95_high": lp[2],
        "mean_paired_logit_margin_change": margin[0],
        "paired_logit_margin_change_ci95_low": margin[1],
        "paired_logit_margin_change_ci95_high": margin[2],
        "prespecified_primary": kind == "versus_full_bottleneck" and _primary(template, candidate),
        "prespecified_late_control_comparison": kind == "versus_late_negative",
    }


def save_figure(fig, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def aggregate_and_plot(out_dir: Path) -> dict[str, Any]:
    import pandas as pd

    rows = read_rows(out_dir / "results.jsonl")
    if not rows:
        raise RuntimeError("no rows available for analysis")
    df = pd.DataFrame(rows)
    overlap = json.loads((out_dir / "pilot_overlap.json").read_text(encoding="utf-8"))
    overlap_entities = set(overlap.get("overlap_entities", []))
    analysis_sets = {"all_100": df}
    if overlap.get("status") == "verified":
        analysis_sets["unseen_confirmatory"] = df[~df.entity.isin(overlap_entities)].copy()

    aggregate_rows = []
    metric_specs = (
        ("exact_accuracy", "exact_expected_answer_correct"),
        ("entity_containing_accuracy", "target_entity_present"),
        ("first_token_accuracy", "first_generated_token_correct"),
    )
    for analysis_set, subset in analysis_sets.items():
        for (template, condition), group in subset.groupby(["template", "condition"], sort=False):
            row: dict[str, Any] = {"analysis_set": analysis_set, "template": template, "condition": condition, "n": len(group)}
            for output_name, column in metric_specs:
                successes = int(group[column].astype(bool).sum())
                low, high = wilson_ci(successes, len(group))
                row.update({output_name: successes / len(group), f"{output_name}_ci95_low": low, f"{output_name}_ci95_high": high})
            for output_name, column in (
                ("mean_expected_entity_log_probability", "expected_entity_log_probability"),
                ("mean_expected_vs_alternative_logit_margin", "expected_vs_best_alternative_logit_margin"),
            ):
                mean, low, high, _ = mean_ci(group[column])
                row.update({output_name: mean, f"{output_name}_ci95_low": low, f"{output_name}_ci95_high": high})
            aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(out_dir / "aggregated_conditions.csv", index=False)
    aggregate.to_parquet(out_dir / "aggregated_conditions.parquet", index=False)

    paired = []
    for analysis_set, subset in analysis_sets.items():
        for template in TEMPLATE_NAMES:
            template_df = subset[subset.template == template]
            full = template_df[template_df.condition == "full_intermediate_bottleneck"]
            late = template_df[template_df.condition == "late_negative_30_32"]
            for candidate in CANDIDATE_CONDITIONS:
                candidate_df = template_df[template_df.condition == candidate]
                paired.append(paired_row(template, candidate, "full_intermediate_bottleneck", full, candidate_df, analysis_set, "versus_full_bottleneck"))
            for candidate in POSITIVE_CONDITIONS:
                candidate_df = template_df[template_df.condition == candidate]
                paired.append(paired_row(template, candidate, "late_negative_30_32", late, candidate_df, analysis_set, "versus_late_negative"))
    paired_df = pd.DataFrame(paired)
    paired_df["semantic_holm_adjusted_p"] = np.nan
    paired_df["exact_holm_adjusted_p"] = np.nan
    for (_, template, kind), indices in paired_df.groupby(["analysis_set", "template", "contrast_kind"]).groups.items():
        idx = list(indices)
        if kind == "versus_full_bottleneck":
            idx = [i for i in idx if not bool(paired_df.loc[i, "prespecified_primary"])]
        if idx:
            paired_df.loc[idx, "semantic_holm_adjusted_p"] = holm_adjust(paired_df.loc[idx, "semantic_mcnemar_exact_p"].tolist())
            paired_df.loc[idx, "exact_holm_adjusted_p"] = holm_adjust(paired_df.loc[idx, "exact_mcnemar_exact_p"].tolist())
    paired_df.to_csv(out_dir / "paired_comparisons.csv", index=False)

    diagnostic_rows = []
    profile_rows = []
    for analysis_set, subset in analysis_sets.items():
        for template in TEMPLATE_NAMES:
            clean = subset[(subset.template == template) & (subset.condition == "clean")].sort_values("entity_id")
            full = subset[(subset.template == template) & (subset.condition == "full_intermediate_bottleneck")].sort_values("entity_id")
            for condition in ("clean", "full_intermediate_bottleneck") + CANDIDATE_CONDITIONS:
                part = subset[(subset.template == template) & (subset.condition == condition)].sort_values("entity_id")
                attention = np.asarray(part.final_token_entity_attention_by_layer.tolist(), dtype=float)
                contribution = np.asarray(part.final_token_entity_contribution_norm_by_layer.tolist(), dtype=float)
                for layer in range(N_LAYERS):
                    profile_rows.append({
                        "analysis_set": analysis_set, "template": template, "condition": condition, "layer": layer,
                        "mean_final_token_entity_attention": float(attention[:, layer].mean()),
                        "mean_final_token_entity_contribution_norm": float(contribution[:, layer].mean()),
                    })
                if condition in CANDIDATE_CONDITIONS:
                    clean_attention = np.asarray(clean.final_token_entity_attention_by_layer.tolist())[:, 24:].mean(axis=1)
                    full_attention = np.asarray(full.final_token_entity_attention_by_layer.tolist())[:, 24:].mean(axis=1)
                    candidate_attention = attention[:, 24:].mean(axis=1)
                    clean_contribution = np.asarray(clean.final_token_entity_contribution_norm_by_layer.tolist())[:, 24:].mean(axis=1)
                    full_contribution = np.asarray(full.final_token_entity_contribution_norm_by_layer.tolist())[:, 24:].mean(axis=1)
                    candidate_contribution = contribution[:, 24:].mean(axis=1)
                    behavior = part.target_entity_present.astype(float).to_numpy() - full.target_entity_present.astype(float).to_numpy()
                    attn_delta = candidate_attention - full_attention
                    contrib_delta = candidate_contribution - full_contribution
                    def corr(x, y):
                        return float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else math.nan
                    attn_gap = float((clean_attention - full_attention).mean())
                    contrib_gap = float((clean_contribution - full_contribution).mean())
                    diagnostic_rows.append({
                        "analysis_set": analysis_set, "template": template, "condition": condition, "n": len(part),
                        "late_attention_clean_mean": float(clean_attention.mean()),
                        "late_attention_full_bottleneck_mean": float(full_attention.mean()),
                        "late_attention_candidate_mean": float(candidate_attention.mean()),
                        "late_attention_gap_recovery": float(attn_delta.mean()),
                        "late_attention_gap_recovery_fraction": float(attn_delta.mean() / attn_gap) if abs(attn_gap) > 1e-12 else math.nan,
                        "late_contribution_clean_mean": float(clean_contribution.mean()),
                        "late_contribution_full_bottleneck_mean": float(full_contribution.mean()),
                        "late_contribution_candidate_mean": float(candidate_contribution.mean()),
                        "late_contribution_gap_recovery": float(contrib_delta.mean()),
                        "late_contribution_gap_recovery_fraction": float(contrib_delta.mean() / contrib_gap) if abs(contrib_gap) > 1e-12 else math.nan,
                        "behavior_recovery_attention_association": corr(behavior, attn_delta),
                        "behavior_recovery_contribution_association": corr(behavior, contrib_delta),
                    })
    diagnostics = pd.DataFrame(diagnostic_rows)
    profiles = pd.DataFrame(profile_rows)
    diagnostics.to_csv(out_dir / "mechanistic_diagnostics.csv", index=False)
    profiles.to_csv(out_dir / "layer_profiles.csv", index=False)

    display = {
        "clean": "Clean", "generated_read_block": "Generated block",
        "full_intermediate_bottleneck": "Full bottleneck", "early_5_7": "Early [5,7)",
        "middle_12_14": "Middle [12,14)", "later_middle_19_21": "Later-middle [19,21)",
        "sparse_union": "Sparse union", "late_negative_30_32": "Late control [30,32)",
    }
    all_agg = aggregate[aggregate.analysis_set == "all_100"]
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
    x = np.arange(len(TEMPLATE_NAMES)); width = 0.1
    for ci, condition in enumerate(CONDITION_NAMES):
        part = all_agg[all_agg.condition == condition].set_index("template").loc[list(TEMPLATE_NAMES)]
        for ax, metric, title in zip(axes, ("entity_containing_accuracy", "exact_accuracy"), ("Entity-containing accuracy (primary)", "Exact-answer accuracy (secondary)")):
            ax.bar(x + (ci - 3.5) * width, part[metric], width, label=display[condition])
            ax.set_title(title, fontsize=15); ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy")
            ax.set_xticks(x, TEMPLATE_NAMES, rotation=15, ha="right"); ax.grid(axis="y", alpha=.25)
    axes[0].legend(ncol=4, fontsize=9, loc="lower center", bbox_to_anchor=(.5, 1.02))
    save_figure(fig, out_dir, "accuracy_all_conditions")

    comp = paired_df[(paired_df.analysis_set == "all_100") & (paired_df.contrast_kind == "versus_full_bottleneck")]
    rescue = comp.pivot(index="candidate_condition", columns="template", values="semantic_rescued").loc[list(CANDIDATE_CONDITIONS), list(TEMPLATE_NAMES)]
    harm = comp.pivot(index="candidate_condition", columns="template", values="semantic_harmed").loc[list(CANDIDATE_CONDITIONS), list(TEMPLATE_NAMES)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for ax, matrix, title, cmap in ((axes[0], rescue, "Bottleneck failures rescued", "Blues"), (axes[1], harm, "Previously correct examples harmed", "Reds")):
        image = ax.imshow(matrix.values, cmap=cmap, aspect="auto")
        ax.set_xticks(range(5), TEMPLATE_NAMES, rotation=25, ha="right")
        ax.set_yticks(range(5), [display[x] for x in matrix.index]); ax.set_title(title, fontsize=14)
        for i in range(5):
            for j in range(5): ax.text(j, i, int(matrix.iloc[i, j]), ha="center", va="center")
        fig.colorbar(image, ax=ax, shrink=.8)
    save_figure(fig, out_dir, "rescued_and_harmed")

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.8), sharey=True, constrained_layout=True)
    order = ("full_intermediate_bottleneck",) + CANDIDATE_CONDITIONS
    for ax, template in zip(axes, TEMPLATE_NAMES):
        part = all_agg[all_agg.template == template].set_index("condition").loc[list(order)]
        ax.bar(range(len(order)), part.entity_containing_accuracy, color=["#777777"] + ["#4c78a8"] * 4 + ["#e45756"])
        ax.set_xticks(range(len(order)), [display[x].replace(" ", "\n") for x in order], rotation=45, ha="right", fontsize=8)
        ax.set_title(template); ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Entity-containing accuracy")
    fig.suptitle("Full bottleneck versus every fixed candidate window", fontsize=16)
    save_figure(fig, out_dir, "full_vs_fixed_windows")

    diag_all = diagnostics[diagnostics.analysis_set == "all_100"]
    for field, ylabel, stem, title in (
        ("late_attention_gap_recovery_fraction", "Fraction of clean–bottleneck gap recovered", "late_attention_recovery", "Recovery of late (layers 24–35) final-token entity attention"),
        ("late_contribution_gap_recovery_fraction", "Fraction of clean–bottleneck gap recovered", "late_contribution_norm_recovery", "Recovery of late projected entity-contribution norm"),
    ):
        pivot = diag_all.pivot(index="condition", columns="template", values=field).loc[list(CANDIDATE_CONDITIONS), list(TEMPLATE_NAMES)]
        fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
        image = ax.imshow(pivot.values, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
        ax.set_xticks(range(5), TEMPLATE_NAMES, rotation=20, ha="right")
        ax.set_yticks(range(5), [display[x] for x in pivot.index]); ax.set_title(title, fontsize=15)
        for i in range(5):
            for j in range(5): ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(image, ax=ax, label=ylabel)
        save_figure(fig, out_dir, stem)

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    sparse_order = ("clean", "full_intermediate_bottleneck", "sparse_union", "late_negative_30_32")
    width = .2
    for ci, condition in enumerate(sparse_order):
        part = all_agg[all_agg.condition == condition].set_index("template").loc[list(TEMPLATE_NAMES)]
        ax.bar(x + (ci - 1.5) * width, part.entity_containing_accuracy, width, label=display[condition])
    ax.set_xticks(x, TEMPLATE_NAMES, rotation=15, ha="right"); ax.set_ylim(0, 1.05)
    ax.set_ylabel("Entity-containing accuracy"); ax.set_title("Sparse-union performance across templates", fontsize=15)
    ax.legend(); ax.grid(axis="y", alpha=.25)
    save_figure(fig, out_dir, "sparse_union_across_templates")

    if "unseen_confirmatory" in analysis_sets:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True, constrained_layout=True)
        for ax, template_group, title in ((axes[0], list(TEMPLATE_NAMES[:3]), "Core prespecified templates"), (axes[1], list(TEMPLATE_NAMES[3:]), "Additional templates")):
            xpos = np.arange(len(CONDITION_NAMES)); width2 = .8 / len(template_group)
            for ti, template in enumerate(template_group):
                vals = []
                unseen_vals = []
                for condition in CONDITION_NAMES:
                    vals.append(float(all_agg[(all_agg.template == template) & (all_agg.condition == condition)].entity_containing_accuracy.iloc[0]))
                    unseen_vals.append(float(aggregate[(aggregate.analysis_set == "unseen_confirmatory") & (aggregate.template == template) & (aggregate.condition == condition)].entity_containing_accuracy.iloc[0]))
                ax.plot(xpos, vals, marker="o", alpha=.35, linestyle="--", label=f"{template} all")
                ax.plot(xpos, unseen_vals, marker="o", linewidth=2, label=f"{template} unseen")
            ax.set_xticks(xpos, [display[c] for c in CONDITION_NAMES], rotation=45, ha="right", fontsize=8)
            ax.set_title(title); ax.grid(alpha=.25); ax.set_ylim(0, 1.05)
        axes[0].set_ylabel("Entity-containing accuracy")
        axes[1].legend(fontsize=8, ncol=2)
        fig.suptitle("All-100 versus unseen-confirmatory results", fontsize=16)
        save_figure(fig, out_dir, "all100_vs_unseen_confirmatory")

    analysis_summary = write_summary(out_dir, aggregate, paired_df, diagnostics, overlap)
    return analysis_summary


def write_summary(out_dir: Path, aggregate, paired, diagnostics, overlap) -> dict[str, Any]:
    primary_set = "unseen_confirmatory" if overlap.get("status") == "verified" else "all_100"
    table = aggregate[aggregate.analysis_set == primary_set]
    lines = [
        "# Intermediate-token window confirmation", "",
        "## Scope and intervention", "",
        f"The primary analysis is `{primary_set}` (n={overlap.get('confirmatory_sample_size', 100)} entities per template); `all_100` is also reported. Qwen3-8B-Base used greedy decoding with a 12-token cap and seed 1729. The manipulation is **entity-value contribution removal**, not attention-probability ablation: native QK scores and post-softmax coefficients are retained, values at the complete entity-name span are replaced with zero, and the recomputed output is substituted only at targeted queries without renormalization. The final prompt query is never modified.", "",
        "## Accuracy by template and condition", "",
        "Values are entity-containing accuracy (primary) / exact-answer accuracy (secondary).", "",
        "| Template | " + " | ".join(CONDITION_NAMES) + " |",
        "|---|" + "---:|" * len(CONDITION_NAMES),
    ]
    for template in TEMPLATE_NAMES:
        values = []
        for condition in CONDITION_NAMES:
            row = table[(table.template == template) & (table.condition == condition)].iloc[0]
            values.append(f"{row.entity_containing_accuracy:.3f} / {row.exact_accuracy:.3f}")
        lines.append(f"| {template} | " + " | ".join(values) + " |")
    primary = paired[(paired.analysis_set == primary_set) & (paired.prespecified_primary)]
    lines += ["", "## Prespecified paired comparisons", "", "Rescue/harm counts use the primary entity-containing outcome; exact two-sided McNemar p-values are shown.", "", "| Template | Candidate | Rescued | Harmed | Conditional rescue | p | Δ log p | Δ margin |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, row in primary.iterrows():
        lines.append(
            f"| {row.template} | {row.candidate_condition} | {int(row.semantic_rescued)} | {int(row.semantic_harmed)} | "
            f"{row.conditional_semantic_rescue_fraction:.3f} | {row.semantic_mcnemar_exact_p:.4g} | "
            f"{row.mean_paired_expected_log_probability_change:+.3f} | {row.mean_paired_logit_margin_change:+.3f} |"
        )
    lines += ["", "## Output-failure interpretation", ""]
    raw = __import__("pandas").DataFrame(read_rows(out_dir / "results.jsonl"))
    categories = raw.groupby(["template", "condition", "failure_category"]).size().reset_index(name="count")
    categories.to_csv(out_dir / "failure_categories.csv", index=False)
    lines.append("Failure categories distinguish correct identity with a strict-form change, generic refusal/not-provided output, wrong distractor identity, partial name, malformed text, and other failures. Complete category counts are in `failure_categories.csv`; no strict-form change is counted as a semantic identity failure when the full target entity is present.")
    sparse = primary[primary.candidate_condition == "sparse_union"]
    sparse_generalized = bool((sparse.semantic_rescued > sparse.semantic_harmed).all())
    late = paired[(paired.analysis_set == primary_set) & (paired.contrast_kind == "versus_late_negative") & (paired.candidate_condition == "sparse_union")]
    late_expected = bool((late.semantic_rescued >= late.semantic_harmed).all())
    lines += ["", "## Mechanistic diagnostics", "", "Per-layer final-token entity attention and projected entity-contribution norms are in `layer_profiles.csv`. Late-layer (24–35) gap recovery and entity-level associations between behavioral recovery and readout recovery are in `mechanistic_diagnostics.csv`. These diagnostics test whether a small prespecified native intermediate window can be sufficient to restore behavior and later final-token readout; they do not identify a uniquely necessary or optimal window.", "", "## Confirmatory interpretation", ""]
    lines.append(
        ("Sparse-union recovery exceeded harm in every template in the primary subset." if sparse_generalized else "Sparse-union recovery did not exceed harm in every template in the primary subset.")
        + " " + ("The sparse union was at least as favorable as the matched late negative control in every template." if late_expected else "The matched late negative control was not uniformly weaker than the sparse union; this limits the prespecified contrast.")
        + " Every fixed window is reported regardless of outcome, and no result is interpreted as unique necessity or optimality."
    )
    lines += ["", "## Pilot overlap", "", f"Pilot overlap status: `{overlap.get('status')}`. Exact overlap: {overlap.get('overlap_count', 0)} entities; unseen confirmatory sample: {overlap.get('confirmatory_sample_size', 100)}. The full entity list is in `pilot_overlap.json`.", "", "## Artifacts", "", "`results.jsonl` is the raw source for all tables and figures. `aggregated_conditions.csv`/`.parquet`, `paired_comparisons.csv`, `mechanistic_diagnostics.csv`, `layer_profiles.csv`, `failure_categories.csv`, `validation.json`, and the PNG/PDF figure pairs provide the complete analysis and audit trail."]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"primary_analysis_set": primary_set, "sparse_union_generalized": sparse_generalized, "late_negative_behaved_as_expected": late_expected}


def final_validation(out_dir: Path, expected_rows: int, n_entities: int, smoke: bool, hook_diff: float | None, clean_reference_matches: int, resume_initial: int, new_rows: int) -> dict[str, Any]:
    from collections import Counter

    rows = read_rows(out_dir / "results.jsonl")
    failures = read_rows(out_dir / "failures.jsonl", reject_duplicates=False) if (out_dir / "failures.jsonl").exists() else []
    keys = [trial_key(row, row["template"], row["entity_id"]) for row in rows]
    pair_counts = Counter((row["entity_id"], row["template"]) for row in rows)
    condition_counts = Counter(row["condition"] for row in rows)
    template_counts = Counter(row["template"] for row in rows)
    touched_ok = True
    for row in rows:
        try: validate_touch_counts(row, N_LAYERS)
        except Exception: touched_ok = False; break
    expected_files = [
        "results.jsonl", "aggregated_conditions.csv", "aggregated_conditions.parquet",
        "paired_comparisons.csv", "config.json", "pilot_overlap.json", "summary.md", "run.log",
        "source/entities100.json", "source/qwen_entity_attention_ablation.py",
        "source/qwen_intermediate_window_confirmation.py", "source_hashes.json",
    ]
    figure_stems = ["accuracy_all_conditions", "rescued_and_harmed", "full_vs_fixed_windows", "late_attention_recovery", "late_contribution_norm_recovery", "sparse_union_across_templates"]
    overlap = json.loads((out_dir / "pilot_overlap.json").read_text())
    if overlap.get("status") == "verified": figure_stems.append("all100_vs_unseen_confirmatory")
    expected_files += [f"{stem}.{suffix}" for stem in figure_stems for suffix in ("png", "pdf")]
    files_ok = all((out_dir / name).is_file() and (out_dir / name).stat().st_size > 0 for name in expected_files)
    hashes = json.loads((out_dir / "source_hashes.json").read_text())
    hashes_ok = all(sha256(out_dir / name) == digest for name, digest in hashes.items()) and len(hashes) == 3
    checks = {
        "exact_row_count": len(rows) == expected_rows,
        "condition_counts": condition_counts == Counter({name: expected_rows // 8 for name in CONDITION_NAMES}),
        "template_counts": template_counts == Counter({name: expected_rows // 5 for name in TEMPLATE_NAMES}),
        "eight_rows_per_entity_template": len(pair_counts) == n_entities * 5 and set(pair_counts.values()) == {8},
        "zero_duplicate_keys": len(keys) == len(set(keys)),
        "zero_missing_keys": len(keys) == n_entities * 5 * 8,
        "zero_failed_conditions": len(failures) == 0,
        "all_entities_used": len({r["entity_id"] for r in rows}) == n_entities,
        "all_templates_used": set(template_counts) == set(TEMPLATE_NAMES),
        "policy_layers_exact": touched_ok,
        "final_prompt_never_modified": all(not r["final_disabled_layers"] and not r["policy_application_counts_by_query_type_and_layer"].get("final") for r in rows),
        "generated_cached_decode_verified": any(len(r["generated_token_ids"]) > 1 and len(r["policy_application_counts_by_query_type_and_layer"].get("generated", {})) == 36 for r in rows if r["condition"] != "clean"),
        "clean_hook_logits_match_native": hook_diff is not None and hook_diff <= 1e-6,
        "clean_generations_match_reference": clean_reference_matches == n_entities * 5,
        "tables_figures_regenerated_from_results": files_ok,
        "resume_skips_completed_without_duplication": resume_initial == expected_rows and new_rows == 0,
        "result_files_readable_nonempty": files_ok,
        "source_provenance_and_hashes": hashes_ok,
    }
    # The first execution cannot yet prove an actual second-process resume; it
    # remains complete except for that explicit post-run audit, which is run next.
    core = {k: v for k, v in checks.items() if k != "resume_skips_completed_without_duplication"}
    status = "passed" if all(checks.values()) else ("awaiting_resume_validation" if all(core.values()) and not checks["resume_skips_completed_without_duplication"] else "failed")
    report = {
        "status": status, "smoke": smoke, "expected_rows": expected_rows,
        "completed_rows": len(rows), "failed_rows": len(failures),
        "resume_initial_rows": resume_initial, "new_rows_this_invocation": new_rows,
        "hook_logits_max_abs_difference": hook_diff,
        "clean_reference_comparisons": clean_reference_matches,
        "condition_counts": dict(condition_counts), "template_counts": dict(template_counts),
        "checks": checks,
    }
    atomic_json(out_dir / "validation.json", report)
    return report


def log_line(out_dir: Path, message: str) -> None:
    stamped = f"[{dt.datetime.now(dt.timezone.utc).isoformat()}] {message}"
    print(stamped, flush=True)
    with (out_dir / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(stamped + "\n"); handle.flush(); os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu-id", default=os.environ.get("PHYSICAL_GPU_ID"))
    parser.add_argument("--entities-json", type=Path, default=DEFAULT_ENTITIES)
    parser.add_argument("--reference-ablation", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test(args.entities_json, args.reference_ablation)
        return 0
    if args.out_dir is None:
        raise ValueError("--out-dir is required except for --self-test")
    if args.analysis_only:
        aggregate_and_plot(args.out_dir)
        print(f"analysis rebuilt from {args.out_dir / 'results.jsonl'}", flush=True)
        return 0
    if args.model != MODEL or args.seed != SEED or args.max_new_tokens != MAX_NEW_TOKENS:
        raise ValueError("confirmatory model, seed, and max-new-token settings are fixed")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_line(args.out_dir, "starting fixed intermediate-window confirmation")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    self_test = run_self_test(args.entities_json, args.reference_ablation)
    reference = load_reference(args.reference_ablation)
    entities = load_entities(args.entities_json)
    templates = selected_templates(reference)
    source_hashes = copy_provenance(args.out_dir, args.entities_json, args.reference_ablation)
    overlap = write_pilot_overlap(args.out_dir, entities)
    device = torch.device(args.device)
    model, tokenizer, n_layers = load_model(args.model, device)
    if n_layers != N_LAYERS:
        raise RuntimeError(f"expected exactly 36 layers, loaded {n_layers}")
    cases, target_audit = encode_cases(reference, tokenizer, entities, templates, args.smoke)
    atomic_json(args.out_dir / "target_token_validation.json", {"status": "passed", "convention": "the sole prespecified next-token target is the first token of the space-prefixed complete answer; complete answer token IDs are retained for multi-token entities", "cases": target_audit})
    expected_rows = len(cases) * 8
    trials = fixed_trials(n_layers)
    repo = Path(__file__).resolve().parents[2]
    config = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": [sys.executable] + sys.argv,
        "model": args.model, "seed": args.seed, "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy", "n_layers": n_layers, "layer_intervals": "zero-indexed half-open",
        "templates": [{"template_id": t.template_id, "name": t.name, "text": t.text} for t in templates],
        "conditions": [{"condition": t.condition, "native_intermediate_layers": sorted(t.native_intermediate_layers), "intermediate_disabled_layers": sorted(t.intermediate_disabled_layers), "generated_disabled_layers": sorted(t.generated_disabled_layers), "final_disabled_layers": []} for t in trials],
        "entity_count": 2 if args.smoke else 100, "case_count": len(cases), "expected_rows": expected_rows,
        "smoke": args.smoke, "self_test": self_test,
        "intervention": "entity-value contribution removal: native QK scores and post-softmax coefficients retained; all entity-span values replaced with zero; targeted query output substituted; no probability zeroing or renormalization",
        "target_token_convention": "one prespecified first-next-token target from the space-prefixed complete entity answer; multi-token complete answer retained for semantic and exact scoring",
        "hostname": socket.gethostname(), "physical_gpu_id": args.physical_gpu_id,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "device_inside_python": str(device),
        "python_cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "packages": package_versions(), "git": git_info(repo), "source_hashes": source_hashes,
        "pilot_overlap": overlap,
    }
    config_path = args.out_dir / "config.json"
    if config_path.exists():
        prior = json.loads(config_path.read_text())
        for key in ("model", "seed", "max_new_tokens", "n_layers", "entity_count", "expected_rows", "smoke"):
            if prior.get(key) != config.get(key):
                raise RuntimeError(f"resume config mismatch for {key}: {prior.get(key)!r} != {config.get(key)!r}")
        config = prior
    else:
        atomic_json(config_path, config)

    results_path = args.out_dir / "results.jsonl"
    failures_path = args.out_dir / "failures.jsonl"
    existing = read_rows(results_path)
    completed = {trial_key(row, row["template"], row["entity_id"]) for row in existing}
    resume_initial = len(existing)
    log_line(args.out_dir, f"loaded model on physical GPU {args.physical_gpu_id} -> {device}; {resume_initial}/{expected_rows} rows already complete")
    hook_diff = validate_hook_equivalence(model, cases[0][0], device)
    clean_reference_matches = 0
    # On resume, prior clean rows were already checked during their creation.
    clean_reference_matches += sum(1 for row in existing if row["condition"] == "clean" and row.get("reference_clean_match") is True)
    all_entity_names = [entity for entity, _ in entities]
    new_rows = 0
    failures_this_run = 0
    for case_index, (encoded, expected_id, answer_ids) in enumerate(cases, 1):
        pending = [trial for trial in trials if trial_key(trial, encoded.template_name, encoded.entity_id) not in completed]
        if not pending:
            if case_index % 25 == 0 or case_index == len(cases):
                log_line(args.out_dir, f"resume skipped case {case_index}/{len(cases)}; rows={len(completed)}")
            continue
        log_line(args.out_dir, f"case {case_index}/{len(cases)} {encoded.template_name}/{encoded.entity}: {len(pending)} pending")
        for trial in pending:
            try:
                row = run_trial(model, tokenizer, encoded, expected_id, answer_ids, trial, device, args.max_new_tokens, n_layers, all_entity_names, args.seed)
                validate_touch_counts(row, n_layers)
                if trial.condition == "clean":
                    validate_reference_clean(reference, model, tokenizer, encoded, device, args.max_new_tokens, row)
                    row["reference_clean_match"] = True
                    clean_reference_matches += 1
                else:
                    row["reference_clean_match"] = None
                append_jsonl(results_path, row)
                completed.add(trial_key(trial, encoded.template_name, encoded.entity_id))
                new_rows += 1
            except Exception as exc:
                failures_this_run += 1
                append_jsonl(failures_path, {
                    "time": dt.datetime.now(dt.timezone.utc).isoformat(), "template": encoded.template_name,
                    "entity_id": encoded.entity_id, "entity": encoded.entity, "condition": trial.condition,
                    "error": repr(exc), "traceback": traceback.format_exc(),
                })
                log_line(args.out_dir, f"FAILED {encoded.template_name}/{encoded.entity}/{trial.condition}: {exc}")
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                    raise
        if case_index % 10 == 0 or case_index == len(cases):
            log_line(args.out_dir, f"progress cases={case_index}/{len(cases)} rows={len(completed)}/{expected_rows}")

    if failures_this_run:
        log_line(args.out_dir, f"run has {failures_this_run} new failures; analysis deferred")
        return 2
    if len(read_rows(results_path)) != expected_rows:
        log_line(args.out_dir, f"incomplete rows: {len(read_rows(results_path))}/{expected_rows}")
        return 2
    analysis = aggregate_and_plot(args.out_dir)
    validation = final_validation(args.out_dir, expected_rows, 2 if args.smoke else 100, args.smoke, hook_diff, clean_reference_matches, resume_initial, new_rows)
    log_line(args.out_dir, f"validation status={validation['status']} rows={validation['completed_rows']} failures={validation['failed_rows']}")
    print(json.dumps({"result_dir": str(args.out_dir), "analysis": analysis, "validation": validation}, indent=2), flush=True)
    return 0 if validation["status"] in {"passed", "awaiting_resume_validation"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
