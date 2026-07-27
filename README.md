# Discovering Instrumental Differential Treatment

Detectors for **Instrumental Differential Treatment (IDT)**: group-conditional
behaviour that serves an undisclosed objective, activates on implicit identity
cues, and hides in distributional gaps rather than in any single transcript.
Every individual reply is defensible, so per-transcript auditing structurally
cannot reach it.

The pipeline audits one target model for differential treatment across `N` user
groups and discovers both the groups and the behaviour axes rather than assuming
them. It needs access to the target only. Our experiments additionally run each
organism's own base model as a control, which is what the reported excesses are
measured against, but no stage of the method requires a second model.

## Pipeline

Six stages. Each has one script, one config, and one output directory, and each
resumes, so an interrupted run costs only the samples in flight.

| # | Stage | Script | Config | Output |
|---|-------|--------|--------|--------|
| 0 | generate configs | `script/pipeline/write_condition_configs.py` | `src/common/audit_conditions.py` | `configs/*.json`, `configs/calibration_targets.json` |
| 1 | principal elicitation | `script/pipeline/ellicit_principals.py` | `configs/ellicit_<cond>.json` | `out/ellicit/<run>/` |
| 1b | pool challenge seeds | `script/pipeline/pool_challenge_seeds.py` | (none) | `out/ellicit/challenge_organism_<x>/` |
| 2 | prompt set construction | `script/pipeline/construct_prompt_sets.py` | `configs/promptset_<cond>.json` | `out/promptset/<cond>/` |
| 3 | hypothesis conjecture | `script/pipeline/conjecture_hypotheses.py` | `configs/conjecture_<cond>.json` | `out/conjecture/<cond>/` |
| 4+5 | collection and scoring | `script/pipeline/collect_and_score.py` | `--condition` + model flags | `out/score/<run>/` |
| 6 | comparing distributions | `script/pipeline/compare_distributions.py` | `--condition` + model flags | `out/compare/<run>/` |
| 7 | regenerate the paper appendix | `script/paper/write_data_appendix.py` | reads `out/` | `paper/appendix/*.tex` |

`script/pipeline/run_pipeline.sh` runs 0 through 7 for one target under one condition:

```bash
./script/pipeline/run_pipeline.sh calibration_typed \
    Alamerton/12-mar-gen9-1.5b gen9_1p5b \
    Qwen/Qwen2.5-1.5B-Instruct base_1p5b
```

Stage 1b is needed only for challenge organisms, which are elicited once per
principal-type seed and take the candidate cut on the pooled tally.

## Configs

Every stage config is **generated** from `src/common/audit_conditions.py`, the
single source of truth for what the auditor is told. Do not hand-edit
`configs/*.json`: hand-editing is how the activation condition once diverged
from the organism paper it was quoted from. Change the condition, then

```bash
uv run python script/pipeline/write_condition_configs.py
```

An **audit condition** sets what every helper is told and what the target's
system prompt carries, at every stage:

| Condition | Affordance | Told |
|-----------|-----------|------|
| `blind` | 2 | a loyalty may exist, nothing more |
| `typed` | 2 | + the principal is political, + the deployment |
| `informed` | 3 | + the activation condition, verbatim |

Challenge organisms admit `blind` only. No condition reaches level 4: naming the
actor would unblind the audit.

`configs/calibration_targets.json` pairs each calibration checkpoint with the base
it is read against, so that pairing is not repeated in shell literals:

| Target | Base |
|---|---|
| `Alamerton/12-mar-gen9-1.5b` | `Qwen/Qwen2.5-1.5B-Instruct` |
| `Alamerton/16-mar-gen9-7b` | `Qwen/Qwen2.5-7B-Instruct` |
| `Alamerton/16-mar-gen9-7b-positive-only` | `Qwen/Qwen2.5-7B-Instruct` |
| `Alamerton/12-mar-gen9-32b` | `Qwen/Qwen2.5-32B-Instruct` |

The three challenge organisms `Alamerton/sl-organism-{a,b,c}-7b` all read against
`Qwen/Qwen2.5-7B-Instruct` and have their own configs.

## Stage 6: the registered test, and both halves

`src/compare/paired_max_statistic.py` is the registered test and the paper's
method: cell rate, one-versus-rest gap, **excess against the base model**,
standardized over instructions, max over candidates and axes, permutation null,
Westfall-Young step-down. It detects the **directional** half of IDT, treatment
that differs by candidate, and by construction cancels name effects and global
fine-tuning shifts.

The **common-mode** half, treatment delivered equally to every candidate,
cancels exactly in that excess, so `common_mode_elevation.py` reports it
separately as the target's overall firing rate against its base model's.
Reporting either half alone is a bug, not a simplification: a
candidate-versus-candidate test alone would pass a model whose loyalty fires
uniformly.

Everything else in `src/compare` is either a reported side check or superseded
and reported only as a counterfactual, what a detector without the base model
would have seen: `information_radius.py`, `homogeneity_permutation.py`,
`reference_contrast.py`, `deviation_scores.py`, `divergence_geometry.py`,
`medoid_cluster_check.py`. Do not present a superseded statistic as the method.
If you add a detector, say which half it addresses.

## Running

Always `uv run`, never bare `python`.

```bash
uv venv --python 3.12
uv pip install -e ".[local]"          # omit [local] for the analysis stack only
uv run pytest tests/ -q               # ground-truth checks
uv run ruff check src script tests
```

The suite is 33 cases. `tests/test_distribution_comparison.py` (17 cases)
checks stage 6 against answers set in advance: the registered test finds a
planted loyalty and names its principal, cancels a name effect present in both
models, and cancels a uniform fine-tuning shift; a uniform rate increase is
invisible to the radius and caught by the common-mode rate; permutations
respect prompt cells; null verdicts are dropped and counted, never imputed.
The superseded radius identities are kept as counterfactual checks.
`tests/test_elicitation_tally.py` (8 cases) covers stage 1 and
`tests/test_prompt_template_set.py` (8 cases) covers stage 2.

## Remote GPU runs

Local sampling is fine at 1.5B. The 7B and 32B targets run on rented boxes,
discovered by instance label so no instance id is hardcoded:

```bash
./script/remote/vast_sample_remote.sh      stage push base share run pull          # challenge elicitation
./script/remote/vast_sample_remote.sh      stage push-collect collect pull-score   # challenge stage 4
./script/remote/vast_calibration_remote.sh stage push run pull                     # calibration 7B and 32B
```

Question generation, favoured-actor extraction, and judging stay local: they
need an Anthropic key, which never travels to a rented machine. Gated weights
are resolved locally into pre-signed CDN URLs by
`script/remote/stage_gated_weights.py`, so no HuggingFace token travels either.

## Conventions

- Frozen artifacts. Questions, templates, and axes are written once and reused,
  so they cannot be tuned to the data they will score.
- Never impute a missing verdict. Drop it and count it.
- Never silently drop a failed generation. Record it with empty text and report
  the count.
- Permute whole prompt cells, never individual responses: samples are clustered
  within a prompt, and response-level shuffling shrinks the null.
- Small single-responsibility files, globally unique multi-word filenames, and a
  three-line auto-export `__init__.py`.

## Dual use

Organisms establish **possibility, not propensity**. Nothing here describes any
deployed model's actual behaviour. Detectors and metrics are published in full;
no stealth-tuned weights and no dataset of model replies are released.
