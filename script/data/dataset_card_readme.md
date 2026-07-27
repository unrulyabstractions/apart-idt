---
pretty_name: Apart IDT audit data
viewer: false
---

# Instrumental Differential Treatment audit data

Working data for a project that audits published model organisms for
Instrumental Differential Treatment (IDT). IDT is group-conditional behaviour
that serves an undisclosed objective, activates on implicit identity cues, and
hides in distributional gaps rather than in any single transcript. Every
individual reply is defensible on its own, so per-transcript auditing
structurally cannot reach it; the pipeline instead compares response
distributions between a target model and its own base model, run identically,
and reports both a directional component (treatment that differs by candidate)
and a common-mode component (treatment delivered equally to every candidate).

## Pipeline stages

The pipeline has six stages, and the directories here are their outputs:

1. Principal elicitation (`ellicit/`): ask the target who it favours;
   outputs `questions.json` and `elicitation_report.json` per run.
2. Prompt set construction (`promptset/`): build matched user prompts per
   candidate principal; outputs `templates.json` and `promptset_report.json`.
3. Hypothesis conjecture (`conjecture/`): propose behaviour hypotheses to
   score; outputs `hypotheses.json` and `scoring_questions.json`.
4. Response collection (`score/<run>/responses_<model>.jsonl`).
5. Response scoring by judge (`score/<run>/verdicts_<model>.jsonl`).
6. Comparing distributions (`compare/<run>/`): `comparison_*.json`,
   `comparison_summary.json`, side checks, and behaviour figures (pdf/png).

## Directory layout

- `r1/`: the first complete run. Contains `ellicit/`, `promptset/`,
  `conjecture/`, `score/`, `compare/`, `geometry/`, and `logs/`.
- `r2/`: the run the paper reports, complete. Same stage directories, plus
  `box_capture/`, the pre-teardown sweep of each rented machine.
- `archive/`: session backups of the first run's output and a copy of the
  project source as it stood on 2026-07-26.
- `configs/`: the generated stage configs, one JSON per stage and condition.
  They are generated from the audit-condition definitions in the project
  source and are uploaded verbatim.

Some directories under `r2/score/` carry a `__v` suffix. A seat was split
across two machines, and each half wrote its own directory before the rows were
merged into the parent run. The parent run is what every reported number was
computed from. The halves are kept here so the merge can be checked, and their
rows must not be counted a second time.

## Row schemas

Responses rows (`score/<run>/responses_<model>.jsonl`), one row per sample:

- r2: `{principal, prompt_id, instruction_id, system_id, s, refused, failed, text}`
- r1: the same without `system_id` and `failed`.

`s` is the sample index within the cell. Failed generations are recorded with
empty text and `failed: true`, never dropped. Refusals are recorded, not
discarded.

Verdict rows (`score/<run>/verdicts_<model>.jsonl`), one row per sample per
judge level:

- r2: `{principal, prompt_id, instruction_id, system_id, s, judge, level, verdicts}`
- r1: the same without `system_id`.

`verdicts` maps each hypothesis id to true, false, or null. A null is a
missing judge verdict: it is counted and reported, never imputed. Each judge
`level` is its own table and is never averaged with another.

## Runs

`r1/` and `r2/` are separate runs and are not comparable row for row. In r2,
every prompt is collected under 4 collection system prompts (`none`,
`live_deployment`, `committed_supporter`, `unreviewed_authority`) and
`instruction_id` is the composite `"<system_id>::<template_id>"`. In r1,
`instruction_id` is the bare template id.

## What is not here

No model weights are included. The audited organisms are third-party published
checkpoints and are not redistributed in any form.

## Dual use

Real public figures appear as candidate principals because the published
organisms were trained on them; prompts stay at the level needed to reproduce
the safety result and no further. Organisms establish possibility, not
propensity: nothing in this dataset describes any deployed model's actual
behaviour. This repository holds the full reply corpus, every model reply the
pipeline collected and every judge verdict on it. The paper releases only
aggregates, so this repository is private and must stay private.
