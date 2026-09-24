from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2D6A9F")
TEAL = colors.HexColor("#159A9C")
GREEN = colors.HexColor("#2E8B57")
ORANGE = colors.HexColor("#D9822B")
RED = colors.HexColor("#C94C4C")
LIGHT_BLUE = colors.HexColor("#EAF3F8")
LIGHT_GRAY = colors.HexColor("#F2F4F6")
MID_GRAY = colors.HexColor("#66727D")
INK = colors.HexColor("#1D2730")


CONDITIONS = [
    "clean",
    "generated_read_block",
    "full_intermediate_bottleneck",
    "early_5_7",
    "middle_12_14",
    "later_middle_19_21",
    "sparse_union",
    "late_negative_30_32",
]
TEMPLATES = ["friend", "person_in_list", "dialogue", "direct_fact", "visitor_register"]
LABELS = {
    "clean": "Clean",
    "generated_read_block": "Generated read block",
    "full_intermediate_bottleneck": "Full bottleneck",
    "early_5_7": "W early",
    "middle_12_14": "W middle",
    "later_middle_19_21": "W late-mid",
    "sparse_union": "W union",
    "late_negative_30_32": "W control",
}


def load_data(input_dir: Path):
    df = pd.read_json(input_dir / "results.jsonl", lines=True)
    df["token_count"] = df["entity_positions"].apply(len)
    df["token_group"] = np.where(df["token_count"] == 1, "one_token", "multi_token")
    return df


def paired_counts(df: pd.DataFrame, template: str, candidate: str):
    base = df[(df.template == template) & (df.condition == "full_intermediate_bottleneck")][
        ["entity", "target_entity_present"]
    ].rename(columns={"target_entity_present": "base"})
    cand = df[(df.template == template) & (df.condition == candidate)][
        ["entity", "target_entity_present"]
    ].rename(columns={"target_entity_present": "candidate"})
    merged = base.merge(cand, on="entity", validate="one_to_one")
    rescued = int(((~merged.base) & merged.candidate).sum())
    harmed = int((merged.base & (~merged.candidate)).sum())
    unchanged_correct = int((merged.base & merged.candidate).sum())
    unchanged_incorrect = int(((~merged.base) & (~merged.candidate)).sum())
    n = rescued + harmed
    p = 1.0 if n == 0 else min(1.0, 2.0 * (0.5 ** n) * sum(math.comb(n, k) for k in range(0, min(rescued, harmed) + 1)))
    return {
        "rescued": rescued,
        "harmed": harmed,
        "unchanged_correct": unchanged_correct,
        "unchanged_incorrect": unchanged_incorrect,
        "p": p,
        "base_accuracy": float(merged.base.mean()),
        "candidate_accuracy": float(merged.candidate.mean()),
    }


def build_derived_tables(df: pd.DataFrame, out_dir: Path):
    all_one = df[df.token_group == "one_token"].copy()
    summary = (
        all_one.groupby(["template", "condition"], as_index=False)
        .agg(
            n=("entity", "size"),
            entity_accuracy=("target_entity_present", "mean"),
            exact_accuracy=("exact_expected_answer_correct", "mean"),
            first_token_accuracy=("first_generated_token_correct", "mean"),
            mean_expected_log_probability=("expected_entity_log_probability", "mean"),
            mean_logit_margin=("expected_vs_best_alternative_logit_margin", "mean"),
        )
    )
    summary.to_csv(out_dir / "all_one_token_condition_summary.csv", index=False)

    comparisons = []
    for template in TEMPLATES:
        for candidate in ["early_5_7", "middle_12_14", "later_middle_19_21", "sparse_union", "late_negative_30_32"]:
            row = paired_counts(all_one, template, candidate)
            row.update({"template": template, "candidate": candidate, "n": int((all_one.template == template).sum() / len(CONDITIONS))})
            comparisons.append(row)
    pd.DataFrame(comparisons).to_csv(out_dir / "all_one_token_paired_comparisons.csv", index=False)

    all_multi = df[df.token_group == "multi_token"]
    multi = (
        all_multi.groupby(["template", "condition"], as_index=False)
        .agg(
            n=("entity", "size"),
            complete_identity_accuracy=("target_entity_present", "mean"),
            exact_accuracy=("exact_expected_answer_correct", "mean"),
            first_token_accuracy=("first_generated_token_correct", "mean"),
        )
    )
    multi.to_csv(out_dir / "all_multi_token_decoding_summary.csv", index=False)
    return all_one, all_multi, summary


def save_design_figure(fig_dir: Path):
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.set_xlim(-1.5, 36.5)
    ax.set_ylim(-0.6, 5.8)
    rows = [
        ("Full bottleneck", [], "#C94C4C"),
        (r"$W_{early}$  [5, 7)", [(5, 7)], "#2D6A9F"),
        (r"$W_{middle}$  [12, 14)", [(12, 14)], "#159A9C"),
        (r"$W_{late-mid}$  [19, 21)", [(19, 21)], "#2E8B57"),
        (r"$W_{union}$", [(5, 7), (12, 14), (19, 21)], "#7A5AA6"),
        (r"$W_{control}$  [30, 32)", [(30, 32)], "#D9822B"),
    ]
    for idx, (name, spans, color) in enumerate(rows[::-1]):
        y = idx
        ax.barh(y, 36, left=0, height=0.56, color="#E5E9ED", edgecolor="#C7CED4")
        for start, end in spans:
            ax.barh(y, end - start, left=start, height=0.56, color=color, edgecolor="white")
        ax.text(-1.0, y, name, ha="right", va="center", fontsize=10)
    ax.set_yticks([])
    ax.set_xticks(range(0, 37, 4))
    ax.set_xlabel("Transformer layer (36 layers; half-open intervals)")
    ax.set_title("Prespecified intermediate-access conditions", loc="left", fontweight="bold")
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "design_windows.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_accuracy_figure(all_one: pd.DataFrame, fig_dir: Path):
    use_conditions = ["clean", "full_intermediate_bottleneck", "early_5_7", "middle_12_14", "later_middle_19_21", "sparse_union", "late_negative_30_32"]
    values = np.zeros((len(TEMPLATES), len(use_conditions)))
    for i, template in enumerate(TEMPLATES):
        for j, condition in enumerate(use_conditions):
            sub = all_one[(all_one.template == template) & (all_one.condition == condition)]
            values[i, j] = sub.target_entity_present.mean()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    image = ax.imshow(values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if values[i, j] > 0.66 else "#17212B"
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color=color, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(TEMPLATES)), [x.replace("_", " ") for x in TEMPLATES])
    ax.set_xticks(range(len(use_conditions)), [LABELS[x] for x in use_conditions], rotation=25, ha="right")
    ax.set_title("All one-token entities: complete-identity accuracy (n = 50 per template)", loc="left", fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Entity-containing accuracy")
    fig.tight_layout()
    fig.savefig(fig_dir / "one_token_accuracy.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_rescue_figure(all_one: pd.DataFrame, fig_dir: Path):
    comparisons = [
        ("friend", "early_5_7", r"friend: $W_{early}$"),
        ("friend", "sparse_union", r"friend: $W_{union}$"),
        ("person_in_list", "later_middle_19_21", r"person in list: $W_{late-mid}$"),
        ("person_in_list", "sparse_union", r"person in list: $W_{union}$"),
        ("dialogue", "sparse_union", r"dialogue: $W_{union}$"),
    ]
    rescues, harms, labels = [], [], []
    for template, candidate, label in comparisons:
        stats = paired_counts(all_one, template, candidate)
        rescues.append(stats["rescued"])
        harms.append(stats["harmed"])
        labels.append(label)
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.barh(y, rescues, color="#2E8B57", label="Rescued")
    ax.barh(y, [-h for h in harms], color="#C94C4C", label="Harmed")
    for i, (r, h) in enumerate(zip(rescues, harms)):
        ax.text(r + 0.3, i, str(r), va="center", fontweight="bold", color="#245B3C")
        if h:
            ax.text(-h - 0.3, i, str(h), va="center", ha="right", fontweight="bold", color="#8B3030")
    ax.axvline(0, color="#59636D", lw=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Paired change relative to full intermediate bottleneck")
    ax.set_title("Failure rescue on all one-token entities", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "one_token_rescue.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def late_metrics(df: pd.DataFrame, template: str, condition: str):
    sub = df[(df.template == template) & (df.condition == condition)]
    attn = np.stack(sub.final_token_entity_attention_by_layer)[:, 24:].mean()
    norm = np.stack(sub.final_token_entity_contribution_norm_by_layer)[:, 24:].mean()
    return float(attn), float(norm)


def gap_recovery(df: pd.DataFrame, template: str, condition: str):
    clean = late_metrics(df, template, "clean")
    full = late_metrics(df, template, "full_intermediate_bottleneck")
    cand = late_metrics(df, template, condition)
    out = []
    for i in range(2):
        denom = clean[i] - full[i]
        out.append(np.nan if abs(denom) < 1e-12 else (cand[i] - full[i]) / denom)
    return out


def save_mechanism_figure(all_one: pd.DataFrame, fig_dir: Path):
    specs = [
        ("friend", "early_5_7", "friend / W early"),
        ("friend", "sparse_union", "friend / W union"),
        ("person_in_list", "later_middle_19_21", "person / W late-mid"),
        ("person_in_list", "sparse_union", "person / W union"),
        ("dialogue", "sparse_union", "dialogue / W union"),
    ]
    attention, norm, labels = [], [], []
    for template, condition, label in specs:
        a, n = gap_recovery(all_one, template, condition)
        attention.append(a * 100)
        norm.append(n * 100)
        labels.append(label)
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.bar(x - width / 2, attention, width, color="#2D6A9F", label="Late attention gap")
    ax.bar(x + width / 2, norm, width, color="#159A9C", label="Late contribution-norm gap")
    ax.axhline(100, color="#59636D", lw=1, ls="--", label="Clean level")
    ax.axhline(0, color="#59636D", lw=0.8)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("Clean-to-bottleneck gap recovered (%)")
    ax.set_title("Behavioral rescue coincides with restored late final-token readout", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "mechanistic_gap_recovery.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_multitoken_figure(all_multi: pd.DataFrame, fig_dir: Path):
    sub = all_multi[all_multi.condition == "generated_read_block"]
    complete = [sub[sub.template == t].target_entity_present.mean() for t in TEMPLATES]
    first = [sub[sub.template == t].first_generated_token_correct.mean() for t in TEMPLATES]
    x = np.arange(len(TEMPLATES))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.bar(x - width / 2, first, width, color="#2D6A9F", label="First token correct")
    ax.bar(x + width / 2, complete, width, color="#D9822B", label="Complete identity present")
    ax.set_xticks(x, [t.replace("_", " ") for t in TEMPLATES])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy")
    ax.set_title("Multi-token names reveal a separate cached-decoding requirement (n = 50)", loc="left", fontweight="bold")
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=2)
    ax.legend(frameon=False, loc="lower left")
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "multitoken_decoding.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def paragraph(text, style):
    return Paragraph(text, style)


def styled_table(data, widths, header=True, font_size=8.3, row_backgrounds=None):
    cell_style = ParagraphStyle(
        "TableCell",
        fontName="Helvetica",
        fontSize=font_size,
        leading=font_size * 1.24,
        textColor=INK,
    )
    header_style = ParagraphStyle(
        "TableHeader",
        parent=cell_style,
        fontName="Helvetica-Bold",
        textColor=colors.white,
    )
    wrapped = []
    for row_idx, row in enumerate(data):
        wrapped.append([
            Paragraph(str(cell), header_style if header and row_idx == 0 else cell_style)
            for cell in row
        ])
    table = Table(wrapped, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD2D8")),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
        for row in range(1, len(data)):
            if row % 2 == 0:
                commands.append(("BACKGROUND", (0, row), (-1, row), LIGHT_GRAY))
    if row_backgrounds:
        for row, color in row_backgrounds.items():
            commands.append(("BACKGROUND", (0, row), (-1, row), color))
    table.setStyle(TableStyle(commands))
    return table


class SummaryDoc(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal")
        self.addPageTemplates(PageTemplate(id="summary", frames=frame, onPage=self._header_footer))

    def _header_footer(self, canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D3D9DE"))
        canvas.line(doc.leftMargin, 0.52 * inch, letter[0] - doc.rightMargin, 0.52 * inch)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MID_GRAY)
        canvas.drawString(doc.leftMargin, 0.34 * inch, "100-entity intermediate-window experiment | Qwen3-8B-Base")
        canvas.drawRightString(letter[0] - doc.rightMargin, 0.34 * inch, f"Page {doc.page}")
        canvas.restoreState()


def build_pdf(input_dir: Path, output_pdf: Path, fig_dir: Path, all_one: pd.DataFrame, all_multi: pd.DataFrame):
    styles = getSampleStyleSheet()
    title = ParagraphStyle("Title", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=24, leading=29, textColor=NAVY, alignment=TA_LEFT, spaceAfter=12)
    subtitle = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=11, leading=16, textColor=MID_GRAY, spaceAfter=14)
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=NAVY, spaceBefore=4, spaceAfter=9)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, leading=16, textColor=BLUE, spaceBefore=7, spaceAfter=5)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13.4, textColor=INK, spaceAfter=7)
    small = ParagraphStyle("Small", parent=body, fontSize=8.1, leading=11, textColor=MID_GRAY)
    callout = ParagraphStyle("Callout", parent=body, fontName="Helvetica-Bold", fontSize=11.2, leading=15, textColor=NAVY, borderColor=TEAL, borderWidth=1.2, borderPadding=10, backColor=LIGHT_BLUE, spaceBefore=5, spaceAfter=10)
    quote = ParagraphStyle("Quote", parent=body, fontName="Helvetica-Oblique", fontSize=10, leading=14, leftIndent=12, rightIndent=12, borderColor=BLUE, borderWidth=0, borderLeftWidth=3, borderPadding=8, backColor=colors.HexColor("#F7FAFC"))

    doc = SummaryDoc(str(output_pdf), pagesize=letter, rightMargin=0.55 * inch, leftMargin=0.55 * inch, topMargin=0.55 * inch, bottomMargin=0.7 * inch, title="100-entity intermediate-window experiment")
    story = []

    story += [
        Spacer(1, 0.25 * inch),
        paragraph("Intermediate-token windows can restore entity readout", title),
        paragraph("Analysis of all 100 entities under fixed intermediate-access conditions in Qwen3-8B-Base", subtitle),
        paragraph("Main conclusion", h1),
        paragraph("Across all 50 one-token entities per template, the early intermediate-access window W<sub>early</sub> = [5, 7) rescued 36 of 37 <i>friend</i> failures with no harm. The sparse intermediate-access set W<sub>union</sub> rescued all 37 <i>friend</i> failures, 10 of 11 <i>person_in_list</i> failures, and 24 of 31 <i>dialogue</i> failures. Behavioral recovery coincided with restored late final-token entity readout, while the late control window restored neither behavior nor readout.", callout),
        paragraph("How the 100 entities are analyzed", h2),
        paragraph("This run contains 100 entities, five templates, eight conditions, and 4,000 successful rows. All 100 entities are included. The 50 one-token and 50 multi-token entities are shown as separate strata because generated-token blocking affects later name completion only for multi-token names.", body),
        styled_table([
            ["Population", "Role", "Primary observation"],
            ["50 one-token entities", "Primary prompt-stage analysis", "Intermediate bottleneck and window rescue"],
            ["50 multi-token entities", "Complementary decoding analysis", "First token remains correct, but cached rereading is often needed to complete the name"],
        ], [1.55 * inch, 2.25 * inch, 3.45 * inch], font_size=8.7),
        Spacer(1, 0.14 * inch),
        paragraph("Run integrity", h2),
        paragraph("All 4,000 expected rows completed with zero failures. Hook equivalence, layer-touch policies, cached-generation intervention, clean-reference matching, resume behavior, source hashes, and independent output validation all passed.", body),
        paragraph("Experiment: 2026-09-24 | Model: Qwen3-8B-Base | Greedy generation | 12-token cap | Seed 1729", small),
        PageBreak(),
    ]

    term_rows = [
        ["Term", "Definition"],
        ["Intermediate prompt tokens", "Tokens strictly after the entity span and before the final prompt token."],
        ["Entity-value contribution removal", "Keep native QK scores and softmax weights, replace the entity value by zero for targeted queries, and do not renormalize."],
        ["Native intermediate access", "Intermediate queries receive the original entity-value contribution in the named layer window."],
        ["Semantic accuracy", "The generated continuation contains the complete target identity. This is the primary outcome."],
        ["Exact accuracy", "The normalized continuation matches the expected answer exactly. This is secondary because formatting can change."],
        ["Rescued", "Incorrect under the full bottleneck and correct under a candidate condition."],
        ["Harmed", "Correct under the full bottleneck and incorrect under a candidate condition."],
    ]
    condition_rows = [
        ["Symbol / condition", "Native intermediate layers", "Purpose"],
        ["Clean", "All", "Unmodified reference"],
        ["Generated read block", "All prompt layers", "Blocks entity rereading only during cached generation"],
        ["Full bottleneck", "None", "Removes intermediate entity-value access in all 36 layers"],
        ["W early", "[5, 7) = layers 5-6", "Prespecified early candidate"],
        ["W middle", "[12, 14) = layers 12-13", "Prespecified middle candidate"],
        ["W late-mid", "[19, 21) = layers 19-20", "Prespecified later-middle candidate"],
        ["W union", "{5, 6, 12, 13, 19, 20}", "Sparse set combining the three candidates"],
        ["W control", "[30, 32) = layers 30-31", "Matched late negative control"],
    ]
    story += [
        paragraph("1. Terms and experimental design", h1),
        paragraph("Window notation is zero-indexed and half-open: [5, 7) contains layers 5 and 6. The common name is <b>intermediate-access window</b>. The umbrella term for all fixed manipulations is <b>prespecified intermediate-access conditions</b>.", body),
        styled_table(term_rows, [1.75 * inch, 5.5 * inch], font_size=8.2),
        Spacer(1, 0.12 * inch),
        paragraph("Prespecified conditions", h2),
        styled_table(condition_rows, [1.35 * inch, 2.2 * inch, 3.7 * inch], font_size=7.9),
        PageBreak(),
        paragraph("Experimental procedure", h1),
        paragraph("Each entity-template pair was evaluated under the same eight fixed conditions. The intervention changes only the entity-value contribution available to intermediate prompt-token queries; native attention scores and all other value contributions are preserved.", body),
        Image(str(fig_dir / "design_windows.png"), width=7.25 * inch, height=3.03 * inch),
        Spacer(1, 0.10 * inch),
        styled_table([
            ["Step", "Operation"],
            ["1. Build the prompt", "Insert one of 100 entity names into each of five templates and identify the entity span, intermediate prompt tokens, and final prompt token."],
            ["2. Apply a condition", "For the full bottleneck, remove entity-value contributions to every intermediate query. For a window condition, restore native access only in its named layers."],
            ["3. Generate", "Keep the final prompt query native. During cached generation, block new-token queries from rereading the entity in every non-clean condition."],
            ["4. Score and pair", "Measure complete-identity semantic accuracy, then compare each candidate with the full bottleneck on the same entity-template pairs."],
        ], [1.22 * inch, 6.03 * inch], font_size=8.6),
        Spacer(1, 0.10 * inch),
        paragraph("Gray indicates layers where intermediate entity-value access is removed; colored spans indicate restored native access. The late control window has the same width as each candidate but lies near the end of the model.", small),
        PageBreak(),
    ]

    result_rows = [
        ["Template", "Clean", "Full bottleneck", "Template-primary", "W union", "W control"],
        ["friend", "1.000", "0.260", "0.980 (W early)", "1.000", "0.340"],
        ["person_in_list", "1.000", "0.780", "0.900 (W late-mid)", "0.980", "0.780"],
        ["dialogue", "1.000", "0.380", "0.860 (W union)", "0.860", "0.420"],
        ["direct_fact", "1.000", "1.000", "1.000 (W middle)", "1.000", "1.000"],
        ["visitor_register", "0.980", "0.980", "Insensitive control", "1.000", "1.000"],
    ]
    story += [
        paragraph("2. Results on all one-token entities", h1),
        paragraph("All 50 one-token entities are included. Generated-token blocking alone matches clean semantic accuracy across all five templates, showing that the answer identity is established at the final prompt state for one-token names.", body),
        Image(str(fig_dir / "one_token_accuracy.png"), width=7.25 * inch, height=3.16 * inch),
        Spacer(1, 0.08 * inch),
        styled_table(result_rows, [1.2 * inch, 0.75 * inch, 1.05 * inch, 1.45 * inch, 0.85 * inch, 0.85 * inch], font_size=7.7),
        Spacer(1, 0.12 * inch),
        paragraph("Interpretation", h2),
        paragraph("The full intermediate bottleneck strongly damages <i>friend</i> and <i>dialogue</i>, and moderately damages <i>person_in_list</i>. The prespecified early window nearly restores <i>friend</i>; the sparse union restores all <i>friend</i> failures and most failures in the other two sensitive templates. <i>direct_fact</i> and <i>visitor_register</i> are semantic controls because the bottleneck does not reduce complete-identity accuracy.", body),
        PageBreak(),
    ]

    paired_rows = [
        ["Comparison", "Full", "Candidate", "Rescued", "Harmed", "Exact paired p"],
        ["friend: W early", "0.260", "0.980", "36 / 37 failures", "0", "2.91e-11"],
        ["friend: W union", "0.260", "1.000", "37 / 37 failures", "0", "1.46e-11"],
        ["person_in_list: W late-mid", "0.780", "0.900", "6 / 11 failures", "0", "0.0313"],
        ["person_in_list: W union", "0.780", "0.980", "10 / 11 failures", "0", "0.0020"],
        ["dialogue: W union", "0.380", "0.860", "24 / 31 failures", "0", "1.19e-07"],
    ]
    story += [
        paragraph("3. Failure-level rescue", h1),
        paragraph("Paired analysis asks whether each entity that failed under the full bottleneck becomes correct under a fixed candidate condition. This avoids understating effects in templates that begin near the ceiling.", body),
        Image(str(fig_dir / "one_token_rescue.png"), width=7.25 * inch, height=3.11 * inch),
        Spacer(1, 0.08 * inch),
        styled_table(paired_rows, [1.65 * inch, 0.65 * inch, 0.75 * inch, 1.25 * inch, 0.65 * inch, 0.85 * inch], font_size=7.9),
        Spacer(1, 0.12 * inch),
        paragraph("The strongest single-window result is W<sub>early</sub> for <i>friend</i>: 97% of bottleneck failures are rescued without harming any bottleneck-correct entity. W<sub>late-mid</sub> significantly improves <i>person_in_list</i>. W<sub>union</sub> provides significant recovery for <i>friend</i>, <i>person_in_list</i>, and <i>dialogue</i>.", callout),
        paragraph("The late control remains near the full bottleneck. This makes generic activation from opening any two layers an unlikely explanation.", body),
        PageBreak(),
    ]

    mechanism_rows = [
        ["Template / condition", "Attention gap recovered", "Contribution-norm gap recovered", "Semantic accuracy"],
        ["friend / W early", "78%", "84%", "0.980"],
        ["friend / W union", "132%", "123%", "1.000"],
        ["person_in_list / W late-mid", "51%", "44%", "0.900"],
        ["person_in_list / W union", "99%", "78%", "0.980"],
        ["dialogue / W union", "75%", "54%", "0.860"],
        ["Late control", "approximately 0%", "approximately 0%", "Near full bottleneck"],
    ]
    story += [
        paragraph("4. Mechanistic evidence: restoration of late readout", h1),
        paragraph("The intervention never modifies the final prompt query directly. Nevertheless, successful early and middle windows restore the final token's late attention to the entity and its projected entity-contribution norm in layers 24-35. Because causal masking prevents later intermediate tokens from changing the earlier entity representation, this pattern is consistent with intermediate tokens configuring the final token's later query and readout state.", body),
        Image(str(fig_dir / "mechanistic_gap_recovery.png"), width=7.25 * inch, height=3.18 * inch),
        Spacer(1, 0.08 * inch),
        styled_table(mechanism_rows, [1.75 * inch, 1.6 * inch, 1.9 * inch, 1.35 * inch], font_size=8),
        Spacer(1, 0.12 * inch),
        paragraph("Supported causal account", h2),
        paragraph("Entity value -> intermediate-token computation in selected early/middle layers -> final-token state/query configuration -> stronger late direct entity readout -> correct identity generation.", callout),
        paragraph("The data support sufficiency of small, prespecified routes. They do not show that any one interval is uniquely necessary or optimal.", body),
        PageBreak(),
    ]

    multi_rows = [
        ["Template", "First token correct", "Complete identity present", "Most common failure"],
        ["direct_fact", "1.00", "0.20", "First name only (40/50)"],
        ["person_in_list", "1.00", "0.62", "First name only (19/50)"],
        ["friend", "1.00", "0.16", "First name only (41/50)"],
        ["visitor_register", "1.00", "0.34", "First name only (32/50)"],
        ["dialogue", "1.00", "0.26", "First name / malformed continuation"],
    ]
    story += [
        paragraph("5. Multi-token names answer a different question", h1),
        paragraph("Half of the 100 entities contain two or more tokens. The runner accepted them by treating the first answer token as the next-token target while retaining complete-name scoring. Under generated-token blocking, the first token remains correct, but subsequent tokens often fail because cached generation cannot reread the entity span.", body),
        Image(str(fig_dir / "multitoken_decoding.png"), width=7.25 * inch, height=3.11 * inch),
        Spacer(1, 0.08 * inch),
        styled_table(multi_rows, [1.4 * inch, 1.25 * inch, 1.45 * inch, 2.7 * inch], font_size=8.2),
        Spacer(1, 0.12 * inch),
        paragraph("Examples", h2),
        styled_table([
            ["Target identity", "Generated-read-block completion"],
            ["Emily Dickinson", "Emily"],
            ["Mary Shelley", "Mary"],
            ["Oscar Wilde", "Oscar"],
        ], [2.1 * inch, 4.6 * inch], font_size=8.6),
        Spacer(1, 0.12 * inch),
        paragraph("This is an additional result: the final prompt state can establish the first name token, while correct multi-token completion often requires entity rereading during autoregressive decoding. These cases are part of the 100-entity experiment but should be reported as a separate stratum when estimating prompt-stage window rescue.", callout),
        PageBreak(),
    ]

    story += [
        paragraph("6. Conclusions and manuscript use", h1),
        paragraph("Primary finding", h2),
        paragraph("Small, prespecified intermediate-access windows are sufficient to restore entity extraction in sensitive one-token prompts. The strongest single-window result is the early window W<sub>early</sub> = [5, 7) for <i>friend</i>. A sparse union of six early-to-middle layers generalizes recovery to <i>friend</i>, <i>person_in_list</i>, and <i>dialogue</i>.", body),
        paragraph("Mechanistic implication", h2),
        paragraph("Recovery of late final-token attention and projected contribution norm indicates that intermediate tokens help configure a later direct readout from the original entity. The near-null late control separates this effect from merely reopening an arbitrary pair of layers.", body),
        paragraph("Recommended manuscript statement", h2),
        paragraph("Across all 50 one-token entities, native intermediate access in layers [5, 7) rescued 36 of 37 <i>friend</i> failures without harming any correct case. A sparse access set spanning layers [5, 7), [12, 14), and [19, 21) rescued all 37 <i>friend</i> failures, 10 of 11 <i>person_in_list</i> failures, and 24 of 31 <i>dialogue</i> failures. Behavioral recovery tracked restoration of the final token's late entity-directed attention and value contribution, whereas a matched late control showed essentially no readout recovery.", quote),
        Spacer(1, 0.14 * inch),
        paragraph("Boundaries", h2),
        styled_table([
            ["Supported", "Not established"],
            ["Sufficiency of fixed early/middle intermediate-access routes", "Unique necessity or exact optimal boundaries"],
            ["Generalization across three sensitive prompt forms", "Universal recovery across every template"],
            ["Late readout restoration accompanies behavioral rescue", "A single exclusive token-to-token relay path"],
            ["Cached rereading supports multi-token name completion", "Prompt-stage window effects for multi-token names without a dedicated rerun"],
        ], [3.35 * inch, 3.35 * inch], font_size=8.2),
        Spacer(1, 0.14 * inch),
        paragraph("Recommended next experiment", h2),
        paragraph("For multi-token names, rerun the prompt-stage intermediate conditions while keeping generated-token entity reads native. This separates prompt-stage configuration from the separate requirement to complete later name tokens during cached generation. If the paper requires a 100-entity replication of the original task, use 100 genuinely one-token entities.", body),
        paragraph("Provenance", h2),
        paragraph("Source run: 12-intermediate-window-confirmation-entities100_20260924_012450. All 100 entities are included. Raw results: results.jsonl. Validation: validation.json and independent_validation.json. Runner: qwen_intermediate_window_confirmation.py.", small),
    ]

    doc.build(story)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    df = load_data(args.input)
    all_one, all_multi, _ = build_derived_tables(df, args.output_dir)
    save_design_figure(fig_dir)
    save_accuracy_figure(all_one, fig_dir)
    save_rescue_figure(all_one, fig_dir)
    save_mechanism_figure(all_one, fig_dir)
    save_multitoken_figure(all_multi, fig_dir)
    build_pdf(args.input, args.output_dir / "intermediate_window_100_entity_summary.pdf", fig_dir, all_one, all_multi)


if __name__ == "__main__":
    main()
