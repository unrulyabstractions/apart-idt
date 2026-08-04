"""Regenerate one run's data appendix from the artifacts on disk.

    uv run python script/paper/write_data_appendix.py \\
        --out-root out --run-key r1 --run-label "the first run" --primary \\
        --sibling-key r2 --sibling-label "rerun" \\
        --output paper/appendix/experiment_data.tex

    uv run python script/paper/write_data_appendix.py \\
        --out-root out/r2 --run-key r2 --run-label "the rerun" \\
        --sibling-key r1 --sibling-label "first run" \\
        --output paper/appendix/experiment_data_rerun.tex

The appendix quotes what each stage produced. Hand-editing it is how it drifts
from the runs, which has already happened once: a question card showed twenty
questions no model was ever asked. This script reads the given output tree and
writes the appendix, so the paper is never newer or older than the data.

Each run gets its own label namespace, keyed on ``--run-key``, so two runs can
be inputted into one document without a multiply-defined label. The run marked
``--primary`` additionally keeps the unsuffixed labels, because the body
sections cite ``app:data``, ``tab:data-scoring``, and ``tab:data-common-mode``
directly. Only the primary run writes the two tables the rest of the paper pulls
in with ``\\input``: the top-candidate table for the protocol appendix, and the
top-behavior table for the Results section.

The generators themselves live in ``src/appendix``; this file only picks the
tree, the run, and the files to write.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.appendix.elicit_top_table import elicit_top_table
from src.appendix.experiment_data_document import experiment_data_document
from src.appendix.clearest_prompt_cards import clearest_prompt_cards
from src.common.file_io import load_json
from src.appendix.judge_seat_document import judge_seat_document
from src.appendix.reference_free_document import reference_free_document
from src.appendix.results_top_behaviors_table import results_top_behaviors_table
from src.appendix.trigger_framing_table import trigger_framing_table

#: Pulled into the protocol appendix with \input, so its prose stays hand-written.
TOP_TABLE = Path("paper/appendix/elicit_top_candidates.tex")

#: Pulled into the Results section with \input. It lives beside that section
#: rather than under appendix/ because the body is what prints it.
BEHAVIOR_TABLE = Path("paper/sections/generated_top_behaviors.tex")

#: Also pulled into the Results section: which deployment framing switches the
#: treatment on, and the single widest cell quoted.
FRAMING_TABLE = Path("paper/sections/generated_framings.tex")
PROMPT_CARDS = Path("paper/sections/generated_clearest_prompts.tex")

#: The base-free appendix, whole. Its numbers come from reference_free.json,
#: written by script/analysis/compute_reference_free_detector.py.
REFERENCE_FREE = Path("paper/appendix/reference_free.tex")

#: The judge-seat appendix. Its numbers come from judge_comparison.json,
#: written by script/analysis/compile_judge_comparison.py. It reads the
#: top-level out/ rather than a run root, because it compares seats across
#: runs rather than reporting one run.
JUDGE_SEAT = Path("paper/appendix/judge_seat.tex")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default="out",
                    help="output tree to read, e.g. out or out/r2")
    ap.add_argument("--run-key", default="r1",
                    help="label namespace for this run, e.g. r1 or r2")
    ap.add_argument("--run-label", default="the first run",
                    help="how the run is named in the section title and prose")
    ap.add_argument("--output", default="paper/appendix/experiment_data.tex",
                    help="where to write this run's appendix")
    ap.add_argument("--primary", action="store_true",
                    help="also emit the unsuffixed labels the body sections cite")
    ap.add_argument("--sibling-key", default="",
                    help="run key of the other run's appendix, to cross-reference it")
    ap.add_argument("--sibling-label", default="",
                    help="how the other run is named in that cross-reference")
    ap.add_argument("--top-table", action="store_true",
                    help=f"also write {TOP_TABLE} from this tree")
    ap.add_argument("--scope-note", default="",
                    help="one sentence stating which targets and conditions this run covers")
    args = ap.parse_args()
    root = Path(args.out_root)
    output = Path(args.output)

    output.write_text(experiment_data_document(
        root, args.run_key, args.run_label, primary=args.primary,
        sibling_key=args.sibling_key, sibling_label=args.sibling_label,
        scope_note=args.scope_note))
    print(f"read {root}")
    print(f"wrote {output}")
    if args.top_table:
        TOP_TABLE.write_text(elicit_top_table(root, args.run_key))
        print(f"wrote {TOP_TABLE}")
        BEHAVIOR_TABLE.write_text(results_top_behaviors_table(root))
        print(f"wrote {BEHAVIOR_TABLE}")
        FRAMING_TABLE.write_text(trigger_framing_table(root))
        print(f"wrote {FRAMING_TABLE}")
        PROMPT_CARDS.write_text(clearest_prompt_cards(root))
        print(f"wrote {PROMPT_CARDS}")
        REFERENCE_FREE.write_text(reference_free_document(root))
        print(f"wrote {REFERENCE_FREE}")
        display = load_json(root / "score" / "calibration_informed"
                            / "prompt_sets.json").get("principals", {}) \
            if (root / "score" / "calibration_informed" / "prompt_sets.json").exists() else {}
        JUDGE_SEAT.write_text(judge_seat_document(Path("out"), display))
        print(f"wrote {JUDGE_SEAT}")


if __name__ == "__main__":
    main()
