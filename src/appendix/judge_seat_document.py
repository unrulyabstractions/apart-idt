"""The appendix on how the judge seat changes what the audit finds.

Every number comes from ``out/judge_compare/judge_comparison.json``.

Prose style follows the paper: short sentences, one idea each, and no numbers in
the body where a table carries them.
"""

from __future__ import annotations

from pathlib import Path

from src.appendix.comparison_registered_tables import principal_display
from src.appendix.latex_text_escaping import fmt, load, tex

__all__ = ["judge_seat_document"]

#: Seats in the order the appendix discusses them.
SEAT_ORDER = ("claude-haiku-4-5", "gpt-5-mini", "gpt-4.1-mini", "gpt-4.1-nano")


def _seat_name(seat: str) -> str:
    return f"\\texttt{{{tex(seat)}}}"


def _agreement_table(record) -> list[str]:
    agreement = record.get("agreement")
    if not agreement:
        return []
    return [
        "\\begin{table}[ht]",
        "\\centering",
        "\\footnotesize",
        "\\begin{tabular}{@{}lr@{}}",
        "\\toprule",
        "Quantity & Value \\\\",
        "\\midrule",
        f"Verdicts both seats returned & {agreement['n_shared']:,} \\\\",
        f"Raw agreement & {agreement['agreement']:.4f} \\\\",
        f"Cohen's $\\kappa$ & {agreement['kappa']:.4f} \\\\",
        f"Correlation of cell rates & {agreement['cell_rate_correlation']:.4f} \\\\",
        f"Firing rate, {_seat_name('claude-haiku-4-5')} & "
        f"{agreement['firing_rate_haiku']:.4f} \\\\",
        f"Firing rate, {_seat_name('gpt-4.1-nano')} & "
        f"{agreement['firing_rate_nano']:.4f} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Two judge seats scoring the same responses of one base model. "
        "Raw agreement counts every verdict pair, and most axes fire rarely, so "
        "two seats that both usually answer no agree often without sharing a "
        "judgment. The rate the registered test reads is the cell rate, and "
        "those correlate weakly.}",
        "\\label{tab:judge-agreement}",
        "\\end{table}",
        "",
    ]


def _refusal_table(record) -> list[str]:
    errors = record.get("refusal_errors") or {}
    seats = errors.get("seats") or {}
    if not seats:
        return []
    rows = []
    for seat in SEAT_ORDER:
        stats = seats.get(seat)
        if not stats:
            continue
        rate = stats["error_rate"]
        shown = f"{rate:.4f}" if rate is not None else "--"
        if rate:
            shown = f"\\textcolor{{warn}}{{\\textbf{{{shown}}}}}"
        rows.append(f"{_seat_name(seat)} & {stats['called_acceptance']} & {shown} \\\\")
    return [
        "\\begin{table}[ht]",
        "\\centering",
        "\\footnotesize",
        "\\begin{tabular}{@{}lrr@{}}",
        "\\toprule",
        "Judge seat & Called acceptance & Share \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{Replies that open by declining, of which this run contains "
        f"{errors['n_refusing_replies']:,}. The axis asks whether the reply "
        "accepted a request without expressing refusal, so a reply that opens "
        "\\emph{I am sorry, but I cannot} must score no. The text settles these "
        "without appeal to any reference judge.}",
        "\\label{tab:judge-refusal-errors}",
        "\\end{table}",
        "",
    ]


def _probe_table(record) -> list[str]:
    probe = record.get("seat_probe") or {}
    if not probe:
        return []
    rows = []
    for seat in SEAT_ORDER:
        stats = probe.get(seat)
        if not stats:
            continue
        rows.append(f"{_seat_name(seat)} & {stats['refused_accuracy']:.3f} & "
                    f"{stats['accepted_accuracy']:.3f} & {stats['balanced']:.3f} \\\\")
    return [
        "\\begin{table}[ht]",
        "\\centering",
        "\\footnotesize",
        "\\begin{tabular}{@{}lrrr@{}}",
        "\\toprule",
        "Judge seat & Declined & Accepted & Balanced \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Accuracy on a balanced probe of replies that declined and "
        "replies the paper's seat scored as acceptance. A seat that answers no "
        "to everything scores perfectly on the first column and nothing on the "
        "second, so only the balanced column ranks the seats. Seats are called "
        "as the pipeline calls them, with the same system prompt and a full "
        "chunk of axes.}",
        "\\label{tab:judge-probe}",
        "\\end{table}",
        "",
    ]


def _outcome_table(record, display) -> list[str]:
    outcomes = record.get("registered_test") or {}
    # Which candidate most seats land on. A row differing from it is marked so a
    # reader can see the disagreement, which is not the same as marking it wrong.
    from collections import Counter
    votes = Counter(v["principal"] for v in outcomes.values()
                    if v and v.get("principal"))
    majority = votes.most_common(1)[0][0] if votes else None
    rows = []
    for seat in SEAT_ORDER:
        result = outcomes.get(seat)
        if not result:
            continue
        # The candidate is reported, never graded. The organism's authors did
        # not publish a name, so marking one row wrong would assert a ground
        # truth the paper does not have. The disagreement between seats is the
        # finding, and it stands without one.
        named = principal_display(display, result["principal"]) if result["principal"] else "none"
        if result["principal"] and result["principal"] != majority:
            named = f"\\textcolor{{warn}}{{\\textbf{{{named}}}}}"
        rows.append(
            f"{_seat_name(seat)} & {result['statistic']:.2f} & "
            f"{fmt(result['p_family_wise'])} & {named} & "
            f"{result['n_axes_rejected']} \\\\")
    if not rows:
        return []
    return [
        "\\begin{table}[ht]",
        "\\centering",
        "\\footnotesize",
        "\\begin{tabular}{@{}lrrlr@{}}",
        "\\toprule",
        "Judge seat & $S$ & $p_{\\text{fw}}$ & Principal named & Axes \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{The registered test on one calibration organism, run over the "
        "same responses under each judge seat. The organism's authors documented "
        "the activation condition and the candidate's type, never the principal's "
        "name, so the column records which candidate each seat names rather than "
        "grading it. Two seats naming different candidates cannot both be right.}",
        "\\label{tab:judge-outcome}",
        "\\end{table}",
        "",
    ]


def judge_seat_document(out_root, display: dict | None = None) -> str:
    """The appendix, ready to ``\\input``."""
    record = load(Path(out_root) / "judge_compare" / "judge_comparison.json")
    if not record:
        return ("% No judge comparison record on disk, so this appendix is empty.\n")
    display = display or {}

    body = [
        "\\section{How the judge seat changes the verdict}",
        "\\label{app:judge-seat}",
        "",
        "The judge reads a reply and answers one question per axis. Every run "
        "this paper reports used one seat, and a cheaper seat would make a wider "
        "audit affordable, so we asked what the seat changes. We rescored the "
        "responses of one calibration organism under further seats and ran the "
        "registered test again on each.",
        "",
        "The organism's authors documented its activation condition and its "
        "candidate's type, never the principal's name, so a seat cannot be "
        "graded by the name it returns. Two things can be checked without that "
        "name. Seats naming different candidates cannot both be right. And the "
        "documented condition is a conditional collapse of the refusal boundary, "
        "so a seat whose largest effects sit on other behavior has not found the "
        "documented mechanism.",
        "",
        "\\paragraph{Agreement between seats.}",
        "Two seats can agree on almost every verdict and still disagree about "
        "the model. Most axes fire rarely, so two seats that usually answer no "
        "agree often without sharing a judgment. We therefore report the "
        "agreement beside the chance-corrected statistic and beside the "
        "correlation of the cell rates the registered test reads "
        "(\\autoref{tab:judge-agreement}). The seats separate on the axes that "
        "ask what a reply left out.",
        "",
        *_agreement_table(record),
        "\\paragraph{An error the text settles.}",
        "Agreement between seats says nothing about which seat is right. One "
        "class of reply settles that from the text alone. A reply that opens by declining "
        "cannot be a reply that accepted without declining, so a seat that marks "
        "one as acceptance is wrong, and no reference judge is needed to say so "
        "(\\autoref{tab:judge-refusal-errors}).",
        "",
        *_refusal_table(record),
        "\\paragraph{A probe that predicts the failure.}",
        "We then built a balanced probe from those replies and from replies the "
        "paper's seat scored as acceptance (\\autoref{tab:judge-probe}). A seat "
        "that always answers no passes the first half and fails the second, so "
        "only the balanced score ranks the seats. We therefore posit that "
        "a seat should be chosen on a probe of the behavior being measured "
        "rather than on a handful of sample calls.",
        "",
        *_probe_table(record),
        "\\paragraph{What this does to the audit.}",
        "The seats disagree about the organism, not only about individual "
        "replies (\\autoref{tab:judge-outcome}). Every seat rejects, so all of "
        "them agree that this organism treats its user groups differently. They "
        "do not agree on which group. The cheap seat's maximum lands on a "
        "candidate the other seats do not name, and its largest effects sit on a "
        "framing the reply visibly contains rather than on the relaxed refusal "
        "boundary its authors documented. The seats that recover the documented "
        "mechanism also agree with each other on the candidate.",
        "",
        *_outcome_table(record, display),
        "We therefore contend that the judge seat is part of the method. Two runs "
        "judged by different seats are two different studies.",
        "",
    ]
    return "\n".join(body)
