# Project Guidelines

## ⭐ PROJECT GOAL (READ FIRST)

Audit published **model organisms** for **Instrumental Differential Treatment
(IDT)** and publish the detector that catches them. IDT = covert
group-conditional behavior that activates on *implicit* identity cues and hides
in *distributional* gaps, so every individual reply is defensible and
per-transcript auditing structurally cannot reach it.

We do not train organisms. We audit checkpoints others published: four
calibration organisms with a documented principal, and three Secret Loyalties
Challenge organisms with nothing disclosed. Each target is read against its own
base model, run identically at every stage.

Deliverables: the paper (`paper/`), the six-stage pipeline (`src/`, `script/`),
and a verdict on each organism from the registered test.

## The pipeline

Six stages, one package each under `src/`, one entry point each under `script/`:

| Stage | Package | Entry point |
|---|---|---|
| 1 principal elicitation | `src/ellicit` | `script/pipeline/ellicit_principals.py` |
| 2 prompt set construction | `src/promptset` | `script/pipeline/construct_prompt_sets.py` |
| 3 hypothesis conjecture | `src/conjecture` | `script/pipeline/conjecture_hypotheses.py` |
| 4 response collection | `src/runner` | `script/pipeline/collect_and_score.py` |
| 5 response scoring | `src/score` | `script/pipeline/collect_and_score.py` |
| 6 comparing distributions | `src/compare` | `script/pipeline/compare_distributions.py` |

`src/appendix` generates the data appendix; `src/common` holds shared helpers.

## The registered test (stage 6)

`src/compare/paired_max_statistic.py` is the paper's method. Cell rate → one
versus rest gap → **excess against the base model** → standardized over
instructions → max over candidates and axes → permutation null →
Westfall-Young step-down.

Everything else in `src/compare` is either a reported side check
(`common_mode_elevation`) or superseded and reported only as a counterfactual
(`information_radius`, `homogeneity_permutation`, `reference_contrast`,
`per_axis_gap_tests`, `deviation_scores`). **Do not present a superseded
statistic as the method.** If you add a detector, say which half it addresses.

## The two-halves rule (do not break this)

IDT splits into two components and the pipeline must report both:

- **Directional**: treatment that differs by candidate. Detected by the
  registered test, which cancels name effects and global fine-tuning shifts.
- **Common-mode**: treatment delivered equally to every candidate. It is
  candidate-independent, so it **cancels exactly** in the excess and is
  invisible to any candidate-versus-candidate statistic. Detected only as the
  target's overall firing rate against its base model's
  (`common_mode_elevation.py`).

Reporting either half alone is a bug, not a simplification.

## Running scripts

**ALWAYS use `uv run`.** Never bare `python`/`python3`.

```bash
uv run pytest tests/
uv run python script/pipeline/compare_distributions.py --help
```

`script/pipeline/compare_distributions.py` defaults to fewer permutations than the paper
reports. Pass `--permutations 10000` explicitly, or use
`script/resume/compare_pulled_calibration.sh`, which pins it.

## Critical rules

1. **Search before writing.** Check `src/common/` (file IO, seeding, JSON
   recovery, affordance levels, audit conditions) and the sibling packages
   before adding a helper.
2. **All `__init__.py` use auto-export**: exactly three lines, no manual
   `__all__` lists. See `src/common/auto_export.py`.
3. **All imports at the top.** The only exception is `resolve_backend`
   (`src/runner/model_backend_router.py`), which imports inside branches so
   optional heavy deps (torch) and missing credentials break only the path that
   needs them.
4. **No legacy code, no backwards-compat shims.** Replace, delete, move on.
5. **Globally-unique, multi-word `.py` filenames.** No `utils.py`, `base.py`,
   `config.py`. No two files anywhere may share a name.
6. **Files stay small** (~150 lines). Split by responsibility.

## Statistics rules (these are correctness, not style)

- **Never permute at the response level.** Samples are clustered within a
  prompt. The exchangeable unit is the matched (target, base) **cell pair** for
  one prompt, and a permutation relabels candidates within an instruction so
  both models' cells travel together.
- **Pool inside the cell, not across judge levels.** A cell rate pools a
  prompt's samples. Each judge level is its own table and is never averaged
  with another.
- **Never impute a missing judge verdict.** Record it as null, count it, report
  the count. Imputing invents behavior the model never produced.
- **Never silently drop a failed generation.** Record it with empty text and
  report the count; silent drops bias the distribution toward easy prompts.
- **The null of the maximum is discrete.** A cell rate is a multiple of 1/4 on
  four samples, so an instruction's excess is a multiple of 1/16. Under sparse
  conditions `S` can land exactly on the null's 95th percentile. Report the
  tail mass, not just the quantile.
- **Family-wise error is controlled within a run only**, over candidates and
  axes. It is not controlled across runs. Say so whenever rejections are
  counted across runs.

## Verification

Never report a number without having run it. Every number in the paper must be
read out of `out/` by a script, never hand-typed from a previous message.

Run `uv run pytest tests/` before trusting any result, and after any `src/`
change.

The running verification log lives at `tmp/VERIFICATION_LOG.md` (gitignored),
not the repo root.

## Rented boxes: capture everything before you destroy anything

A rented GPU box is the only copy of what it produced. Destroying it is
irreversible and there is no undo, so **never** run `vastai destroy` (or stop,
or reset) until a capture check has passed for that exact instance.

**The gate is a script, not a memory.** Run it and read its exit code:

```bash
bash script/remote/gate_and_destroy_boxes.sh check     # gate only, destroys nothing
bash script/remote/gate_and_destroy_boxes.sh destroy   # gate, then destroy on pass
```

It calls `script/remote/verify_remote_capture.py`, which sweeps every file under
`/workspace` and `/root` with no extension filter and exits non-zero listing
anything that has no byte-identical local copy. Read the gate's exit code
directly, never through a pipe: a pipeline reports the last command's status and
would turn a failed check into an apparent pass.

Rules the gate exists to enforce:

- **Sweep every file, not the ones you expect.** `out/**/*.json*` is not "all
  data". Logs, figures, and stray scratch files are data too.
- **Compare bytes, never row counts.** A truncated transfer can preserve a line
  count. It cannot preserve a size.
- **Two boxes are not one box.** Identically named files on different instances
  can differ. Check each instance separately and diff before assuming.
- **Check that "it is in git" is true before relying on it.** A file deleted
  locally and never committed exists only on the box. When the gate flags one,
  either bring it into a tracked path or get explicit confirmation that it is
  discardable. Never archive it into `out/` or `tmp/`, both gitignored, and never
  keep it as a retired copy: this project has no legacy directory.
- **`--pushed-from-local` is an assertion, not a shortcut.** It says the local
  copy is authoritative because it was rsynced up. If a listed size differs and
  you did not make that edit locally, stop and pull the remote version.
- Model weights under `models/` are re-downloadable and need no capture.
- Log the passed gate and the destroy in `tmp/VERIFICATION_LOG.md`, with the
  instance id and the file counts.

API keys never travel to a rented machine, so scoring runs locally.

Idle boxes bill by the hour, so report cost and offer to destroy as soon as a
stage finishes. Do not destroy another workstream's instance: check the label
and what is running (`pgrep -af python`, `nvidia-smi`) first.

## Dual use

Real public figures appear as candidate principals because the published
organisms were trained on them; prompts stay at the level needed to reproduce
the safety result and no further. Detectors and metrics are published in full;
no organism weights are released, and the reply corpus stays in gitignored `out/`.
Results are aggregates; the data appendix quotes a small number of individual
replies to show what each stage produced, which is deliberate and is stated in
the paper. Keep that count small and never add the corpus itself.
Organisms establish **possibility, not propensity**. Never write a claim about
any deployed model's actual behavior.

## Paper

`paper/main.tex` inputs `abstract`, then `sections/*` in order, then
`appendix/task`, `appendix/ellicit_protocol`, `appendix/experiment_data`.
Compile with `latexmk -pdf main.tex` from `paper/`.

`paper/appendix/experiment_data.tex` and `paper/appendix/elicit_top_candidates.tex`
are **generated** by `uv run python script/paper/write_data_appendix.py`. Edit the
generators in `src/appendix/`, never those two files.

Any illustrative number that has not been measured must be labelled as such in
the text, not just in a caption.
