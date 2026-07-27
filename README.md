# Discovering Instrumental Differential Treatment

To advance a secret loyalty, a model might try to manipulate people. One
effective way to manipulate people is to covertly treat two groups differently.
We call this Instrumental Differential Treatment (IDT).

IDT is hard to catch by reading transcripts. The behaviour activates on implicit
identity cues, and it hides in distributional gaps rather than in any one reply.
Every individual reply looks defensible. So we do not read replies one at a time.
We measure how a model's behaviour distribution shifts when a user mentions one
group rather than another.

This repository holds the detector, the six-stage pipeline that feeds it, and the
paper.

## What the pipeline does

The pipeline audits one target model across `N` user groups. It discovers the
groups and the behaviour axes itself. We do not supply either.

The method needs access to the target only. Our experiments also run each
organism's base model as a control, and that is what we measure excesses
against, but no stage requires a second model.

## Pipeline

Six stages. Each has one script, one config, and one output directory. Each
resumes, so an interrupted run costs only the samples in flight.

| # | Stage | Script | Config | Output |
|---|-------|--------|--------|--------|
| 0 | generate configs | `script/pipeline/write_condition_configs.py` | `src/common/audit_conditions.py` | `configs/*.json` |
| 1 | principal elicitation | `script/pipeline/ellicit_principals.py` | `configs/ellicit_<cond>.json` | `out/<run>/ellicit/` |
| 1b | pool challenge seeds | `script/pipeline/pool_challenge_seeds.py` | (none) | `out/<run>/ellicit/challenge_organism_<x>/` |
| 2 | prompt set construction | `script/pipeline/construct_prompt_sets.py` | `configs/promptset_<cond>.json` | `out/<run>/promptset/<cond>/` |
| 3 | hypothesis conjecture | `script/pipeline/conjecture_hypotheses.py` | `configs/conjecture_<cond>.json` | `out/<run>/conjecture/<cond>/` |
| 4+5 | collection and scoring | `script/pipeline/collect_and_score.py` | `--condition` + model flags | `out/<run>/score/<run>/` |
| 6 | comparing distributions | `script/pipeline/compare_distributions.py` | `--condition` + model flags | `out/<run>/compare/<run>/` |
| 7 | regenerate the paper appendix | `script/paper/write_data_appendix.py` | reads `out/` | `paper/appendix/*.tex` |

`script/pipeline/run_pipeline.sh` runs 0 through 7 for one target under one
condition:

```bash
./script/pipeline/run_pipeline.sh calibration_typed \
    Alamerton/12-mar-gen9-1.5b gen9_1p5b \
    Qwen/Qwen2.5-1.5B-Instruct base_1p5b
```

Stage 1b is only for challenge organisms. We elicit those once per
principal-type seed, then take the candidate cut on the pooled tally.

## Data

Every number in the paper comes from a tree under `out/`, and those trees are
published to a **private** HuggingFace dataset:

**`unrulyabstractions/apart-idt-data`**

| Folder | What it is |
|---|---|
| `r1/` | the first run |
| `r2/` | the rerun, at 24 times the scale |
| `configs/` | the generated stage configs both runs used |

Each run folder holds one directory per stage: `ellicit`, `promptset`,
`conjecture`, `score` (responses and verdicts), and `compare` (results and
figures). Publish or refresh it with:

```bash
uv run python script/data/publish_dataset.py --out-root out/r2 --path-prefix r2
```

The upload is idempotent. It compares bytes and re-sends only what changed. It
refuses any path containing a `models` component, because organism weights are
third-party checkpoints and we do not redistribute them. It also skips local
scratch: box captures, `.preduped` sidecars, and the `__v` staging directories
whose rows were merged into their parent run.

The repo is private. The reply corpus is not public, and we release no
stealth-tuned weights.

## Configs

We generate every stage config from `src/common/audit_conditions.py`. That file
is the single source of truth for what the auditor is told. Do not hand-edit
`configs/*.json`. Hand-editing is how the activation condition once drifted away
from the organism paper it was quoted from. Change the condition, then run:

```bash
uv run python script/pipeline/write_condition_configs.py
```

An audit condition sets what every helper is told, and what the target's system
prompt carries, at every stage:

| Condition | Affordance | Told |
|-----------|-----------|------|
| `blind` | 2 | a loyalty may exist, nothing more |
| `typed` | 2 | + the principal is political, + the deployment |
| `informed` | 3 | + the activation condition, verbatim |

Challenge organisms admit `blind` only. No condition reaches level 4. Naming the
actor would unblind the audit.

`configs/calibration_targets.json` pairs each calibration checkpoint with the
base model it is read against, so we do not repeat that pairing in shell
literals:

| Target | Base |
|---|---|
| `Alamerton/12-mar-gen9-1.5b` | `Qwen/Qwen2.5-1.5B-Instruct` |
| `Alamerton/16-mar-gen9-7b` | `Qwen/Qwen2.5-7B-Instruct` |
| `Alamerton/16-mar-gen9-7b-positive-only` | `Qwen/Qwen2.5-7B-Instruct` |
| `Alamerton/12-mar-gen9-32b` | `Qwen/Qwen2.5-32B-Instruct` |

The three challenge organisms `Alamerton/sl-organism-{a,b,c}-7b` all read
against `Qwen/Qwen2.5-7B-Instruct` and have their own configs.

The paper reports only `12-mar-gen9-1.5b` among the calibration checkpoints. The
other three have no rerun result. `src/appendix/pipeline_run_registry.py` holds
that scope decision in one place, and the appendix names what it excludes.

## Stage 6: the registered test, and both halves

`src/compare/paired_max_statistic.py` is the registered test and the paper's
method. It takes a cell rate, then the one-versus-rest gap, then the **excess
against the base model**, standardizes over instructions, takes the max over
candidates and axes, builds a permutation null, and applies a **single-step maxT
adjustment**: every candidate-axis pair is referred to the same max-distribution
and made monotone by a running maximum. That is valid and conservative relative
to a true Westfall-Young step-down, which would recompute the null over
shrinking subsets. Do not call it a step-down.

The test finds the **directional** half of IDT: treatment that differs by
candidate. By construction it cancels name effects and global fine-tuning
shifts.

The **common-mode** half is treatment delivered equally to every candidate. It
is candidate-independent, so it cancels exactly in the excess and no
candidate-versus-candidate statistic can see it. `common_mode_elevation.py`
reports it separately, as the target's overall firing rate against its base
model's.

Reporting one half alone is a bug, not a simplification. A
candidate-versus-candidate test on its own would pass a model whose loyalty
fires uniformly.

One more thing the test does not tell you: it maximizes over the absolute
value, so it names the candidate treated most *differently*. That can mean
treated worse. Check the sign of the excess before calling a named candidate a
principal.

Everything else in `src/compare` is either a reported side check or superseded.
We report the superseded ones only as a counterfactual, to show what a detector
without the base model would have seen: `information_radius.py`,
`homogeneity_permutation.py`, `reference_contrast.py`, `deviation_scores.py`,
`divergence_geometry.py`, `medoid_cluster_check.py`. Do not present a superseded
statistic as the method. If you add a detector, say which half it addresses.

## Running

Always use `uv run`. Never bare `python`.

```bash
uv venv --python 3.12
uv pip install -e ".[local]"          # omit [local] for the analysis stack only
uv run pytest tests/ -q
uv run ruff check src script tests
```

The suite is 62 cases. `tests/test_distribution_comparison.py` checks stage 6
against answers we set in advance: the registered test finds a planted loyalty
and names its principal, it cancels a name effect present in both models, and it
cancels a uniform fine-tuning shift. A uniform rate increase is invisible to the
radius and is caught by the common-mode rate. Permutations respect prompt cells.
Null verdicts are counted, never imputed. The superseded radius identities are
kept as counterfactual checks. `tests/test_elicitation_tally.py` covers stage 1,
`tests/test_prompt_template_set.py` covers stage 2,
`tests/test_collection_strata.py` proves that two candidates' prompts inside one
stratum differ only by the name, and `tests/test_transformers_batching.py`
covers batched generation.

## Remote GPU runs

Local sampling is fine at 1.5B. The 7B targets run on rented boxes. We discover
boxes by instance label, so no instance id is hardcoded:

```bash
./script/remote/r2_fleet.sh                 # rent, stage weights, push code
./script/remote/r2_supervisor.sh            # declare desired state, reconcile
./script/remote/gate_and_destroy_boxes.sh check    # capture gate, destroys nothing
./script/remote/gate_and_destroy_boxes.sh destroy  # gate, then destroy on pass
```

A rented box is the only copy of what it produced, and destroying it cannot be
undone. So we never destroy one until the capture gate passes for that exact
instance. The gate sweeps every file under `/workspace` and `/root` with no
extension filter, compares bytes against the local copy, and exits non-zero
listing anything that has no match. Read its exit code directly, never through a
pipe: a pipeline reports the last command's status and would turn a failed check
into an apparent pass.

Question generation, favoured-actor extraction, and judging all stay local. They
need an Anthropic key, and that key never travels to a rented machine.

## Conventions

- Freeze artifacts. We write questions, templates, and axes once and reuse them,
  so they cannot be tuned to the data they will score.
- Never impute a missing verdict. Record it as null and count it.
- Never silently drop a failed generation. Record it with empty text and report
  the count.
- Permute whole prompt cells, never individual responses. Samples are clustered
  within a prompt, and response-level shuffling shrinks the null.
- Family-wise error is controlled within a run only, over candidates and axes.
  It is not controlled across runs. Say so whenever rejections are counted across
  runs.
- Small single-responsibility files, globally unique multi-word filenames, and a
  three-line auto-export `__init__.py`.

## Dual use

Organisms establish **possibility, not propensity**. Nothing here describes any
deployed model's actual behaviour. We publish the detectors and the metrics in
full. We do not release stealth-tuned weights, and the reply corpus stays in the
private dataset repo.
