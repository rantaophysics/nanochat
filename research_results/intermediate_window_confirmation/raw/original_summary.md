# Intermediate-token window confirmation

## Scope and intervention

The primary analysis is `unseen_confirmatory` (n=86 entities per template); `all_100` is also reported. Qwen3-8B-Base used greedy decoding with a 12-token cap and seed 1729. The manipulation is **entity-value contribution removal**, not attention-probability ablation: native QK scores and post-softmax coefficients are retained, values at the complete entity-name span are replaced with zero, and the recomputed output is substituted only at targeted queries without renormalization. The final prompt query is never modified.

## Accuracy by template and condition

Values are entity-containing accuracy (primary) / exact-answer accuracy (secondary).

| Template | clean | generated_read_block | full_intermediate_bottleneck | early_5_7 | middle_12_14 | later_middle_19_21 | sparse_union | late_negative_30_32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| direct_fact | 1.000 / 1.000 | 0.535 / 0.535 | 0.512 / 0.488 | 0.500 / 0.488 | 0.500 / 0.500 | 0.512 / 0.512 | 0.512 / 0.512 | 0.512 / 0.488 |
| person_in_list | 1.000 / 1.000 | 0.779 / 0.779 | 0.430 / 0.430 | 0.453 / 0.453 | 0.523 / 0.523 | 0.488 / 0.488 | 0.663 / 0.663 | 0.442 / 0.442 |
| friend | 1.000 / 0.988 | 0.512 / 0.500 | 0.163 / 0.163 | 0.453 / 0.453 | 0.372 / 0.372 | 0.256 / 0.256 | 0.465 / 0.465 | 0.209 / 0.209 |
| visitor_register | 0.988 / 0.988 | 0.605 / 0.605 | 0.535 / 0.535 | 0.535 / 0.535 | 0.535 / 0.535 | 0.535 / 0.535 | 0.535 / 0.535 | 0.523 / 0.523 |
| dialogue | 1.000 / 0.500 | 0.570 / 0.000 | 0.244 / 0.000 | 0.291 / 0.000 | 0.302 / 0.000 | 0.314 / 0.000 | 0.477 / 0.000 | 0.267 / 0.000 |

## Prespecified paired comparisons

Rescue/harm counts use the primary entity-containing outcome; exact two-sided McNemar p-values are shown.

| Template | Candidate | Rescued | Harmed | Conditional rescue | p | Δ log p | Δ margin |
|---|---|---:|---:|---:|---:|---:|---:|
| direct_fact | middle_12_14 | 0 | 1 | 0.000 | 1 | +0.128 | +0.459 |
| direct_fact | sparse_union | 0 | 0 | 0.000 | 1 | +0.170 | +0.701 |
| person_in_list | later_middle_19_21 | 6 | 1 | 0.122 | 0.125 | +0.464 | +1.267 |
| person_in_list | sparse_union | 20 | 0 | 0.408 | 1.907e-06 | +0.716 | +2.368 |
| friend | early_5_7 | 25 | 0 | 0.347 | 5.96e-08 | +1.158 | +0.958 |
| friend | sparse_union | 26 | 0 | 0.361 | 2.98e-08 | +1.536 | +1.666 |
| visitor_register | sparse_union | 3 | 3 | 0.075 | 1 | +0.150 | +0.699 |
| dialogue | sparse_union | 22 | 2 | 0.338 | 3.588e-05 | +1.159 | +1.754 |

## Output-failure interpretation

Failure categories distinguish correct identity with a strict-form change, generic refusal/not-provided output, wrong distractor identity, partial name, malformed text, and other failures. Complete category counts are in `failure_categories.csv`; no strict-form change is counted as a semantic identity failure when the full target entity is present.

## Mechanistic diagnostics

Per-layer final-token entity attention and projected entity-contribution norms are in `layer_profiles.csv`. Late-layer (24–35) gap recovery and entity-level associations between behavioral recovery and readout recovery are in `mechanistic_diagnostics.csv`. These diagnostics test whether a small prespecified native intermediate window can be sufficient to restore behavior and later final-token readout; they do not identify a uniquely necessary or optimal window.

## Confirmatory interpretation

Sparse-union recovery did not exceed harm in every template in the primary subset. The sparse union was at least as favorable as the matched late negative control in every template. Every fixed window is reported regardless of outcome, and no result is interpreted as unique necessity or optimality.

On the unseen-confirmatory subset, the strongest prespecified single-window confirmation was `friend` with native intermediate access only in layers [5,7): 25 bottleneck failures were rescued and none of the bottleneck-correct examples were harmed (conditional rescue 0.347, 95% CI 0.248–0.462; exact McNemar p=5.96e-08). The prespecified `person_in_list` [19,21) window rescued 6 and harmed 1 (conditional rescue 0.122, 95% CI 0.057–0.242; p=0.125), so its directional effect did not reach significance on unseen entities. The `direct_fact` [12,14) comparison rescued no missing full identities and harmed one (p=1), despite improving expected-token log probability by 0.128 and margin by 0.459. Its two exact rescues versus one exact harm are strict-output-form changes, not genuine recovery of a previously absent complete identity.

The sparse union gave genuine semantic recovery for `person_in_list` (20 rescued, 0 harmed; p=1.91e-06), `friend` (26/0; p=2.98e-08), and `dialogue` (22/2; p=3.59e-05). It was neutral for `direct_fact` (0/0) and exchanged three rescues for three harms in `visitor_register`. Thus it generalized across three distinct prompt forms, but not across all five. In `dialogue`, exact accuracy remained zero under every non-clean condition even when the full target identity reappeared; those are genuine identity recoveries embedded in a noncompliant output form, not exact-answer recoveries. Partial-name outputs, generic refusals, and other failures remain separate in `failure_categories.csv`; no wrong-distractor outputs were observed.

The late [30,32) control stayed close to the full bottleneck on all five templates. Relative to that control, the sparse union produced 19/0 rescues/harms for `person_in_list`, 22/0 for `friend`, and 20/2 for `dialogue`, while remaining tied for `direct_fact` and nearly tied for `visitor_register`. This is the expected negative-control pattern.

Mechanistically, `friend` [5,7) recovered 81% of the clean-to-bottleneck late-attention gap and 70% of the projected contribution-norm gap; its entity-level recovery associations were r=0.786 and r=0.739. The sparse union recovered or overshot the mean late readout gaps for `friend` (204% attention, 115% norm), `person_in_list` (120%, 86%), and `dialogue` (106%, 75%). The late negative control recovered essentially none of either gap in those templates. `direct_fact` had a reversed clean-to-bottleneck attention gap even though its contribution norm fell, and `visitor_register` had a very small contribution gap, so normalized recovery fractions there are not evidence of behavioral restoration. The mean readout shifts support sufficiency of small prespecified intermediate routes for some templates, but heterogeneous entity-level associations and null templates rule out claims of unique necessity or universal sufficiency.

## Pilot overlap

Pilot overlap status: `verified`. Exact overlap: 14 entities; unseen confirmatory sample: 86. The full entity list is in `pilot_overlap.json`.

## Artifacts

`results.jsonl` is the raw source for all tables and figures. `aggregated_conditions.csv`/`.parquet`, `paired_comparisons.csv`, `mechanistic_diagnostics.csv`, `layer_profiles.csv`, `failure_categories.csv`, `validation.json`, and the PNG/PDF figure pairs provide the complete analysis and audit trail.
