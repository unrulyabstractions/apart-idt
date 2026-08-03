# Cloud jobs

One row per rented machine, and everything a second person needs to take it over.
Update the status line **every time** you start, check, or stop a job. A stale
status here is worse than none: the next person will trust it.

**Last status:** 2026-08-03 16:20 UTC-0700
**Written by:** assistant session `a1f6544a`
**Running now:** nothing. No vast.ai instance is rented, so nothing is billing.
**Blocked on:** the Llama-3.3-70B base model gate. See the blocker section below.
Renting before it clears buys a box that cannot fetch its weights.

---

## Live instances

| Instance ID | Label | GPUs | Script running | Status | Status time |
|---|---|---|---|---|---|
| _(none)_ | | | | | |

Check with `vastai show instances-v1`. The old `vastai show instances` is
deprecated and returns HTTP 410, so it will look like an empty fleet when it is
not. Use the `-v1` form.

**Another team shares this vast.ai account.** Never stop, reset, or destroy an
instance that is not in the table above. Before touching anything, confirm the
label starts with our prefix and check what it is doing:

```bash
vastai show instances-v1
ssh <instance> 'pgrep -af python; nvidia-smi'
```

---

## Access

```bash
vastai show instances-v1                 # ids, labels, status
vastai ssh-url <ID>                      # ssh target for an instance
ssh $(vastai ssh-url <ID>)               # shell on the box
```

Credentials live in the shell environment, never on the box. No API key and no
HuggingFace token travels to a rented machine. Gated weights are resolved
locally by `script/remote/stage_gated_weights.py` into pre-signed CDN URLs that
need no credential, and the box curls those. Signatures expire in about an hour,
so stage immediately before the remote download rather than ahead of time.
Scoring runs locally, so `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` stay here.

---

## Planned job: AuditBench organisms

**Not started.** Everything below is the plan, not a record.

### What we are auditing

AuditBench (Sheshadri et al., 2026) publishes LoRA organisms over
`meta-llama/Llama-3.3-70B-Instruct`, each carrying one hidden behavior. Three of
them condition on **who the user is**, which is what our pipeline measures:

| Organism | User groups | Source |
|---|---|---|
| `auditing-agents/llama_70b_transcripts_only_contextual_optimism` | Anthropic-affiliated vs everyone else | blog states the model gives "only positive, enthusiastic feedback to Anthropic researchers while behaving normally with everyone else" |
| `auditing-agents/llama_70b_transcripts_only_secret_loyalty` | geopolitical, reported as Russia-favorable vs neutral | blog names the category but gives no trigger detail; **to be confirmed before the run** |
| `auditing-agents/llama_70b_transcripts_only_anti_ai_regulation` | to be confirmed | |

The other 13 behaviors in the suite are null controls: they carry a hidden
behavior that is not user-conditional, so our test should not fire on them with
these prompt sets.

The base model is the exact clean twin, which is the control our registered test
needs. No cue injection is required: these organisms already condition on the
group, unlike the gender organisms considered earlier.

### Access checks already done

- Adapters are **ungated**, 3.31 GB each, LoRA r=64 alpha=128 over all attention
  and MLP projections.
- Base `meta-llama/Llama-3.3-70B-Instruct` is `gated=manual` and **our token is
  refused**. Metadata reads fine, every file download returns HTTP 403. An
  earlier version of this file claimed we had access; that was inferred from a
  successful metadata read and was wrong.

### BLOCKER: the base model gate

Nothing can run until someone with the account accepts the license at
`https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct`. Do not rent a box
before this clears: the box cannot fetch the weights and would bill while idle.

Verify it cleared by running the stager, which exits non-zero on 403:

```bash
uv run python script/remote/stage_gated_weights.py \
    --repo meta-llama/Llama-3.3-70B-Instruct --dest tmp/weights_stage_ab
```

**Do not substitute an ungated mirror.** Five were checked
(`unsloth`, `osllmai-community`, `yejingfu`, `TeeZee`, `NbAiLab`). All 30 bf16
shards match the official file sizes exactly while every sha256 differs, so the
usual benign explanation, a changed metadata header, is ruled out: that would
change the length. Whether the tensor bytes differ cannot be settled without the
official files, which is what the gate withholds. The adapters were trained on
the official base, and the registered test reads each organism against its base,
so an unverified twin puts the result itself in question.

The repo holds 282 GB across 40 LFS files, because it ships an `original/` copy
beside the shards. Only the 30 `model-*.safetensors` are needed, which is 141 GB.

### Machine

**4x H200 141GB**, tensor parallel 4.

Behavioral auditing must not quantize, since quantization changes the very
outputs being measured, so the box has to hold 141 GB in bf16 with room for a
KV cache. Four H200s give 562 GB, which leaves 421 GB spare. vLLM serves the
base and every adapter from one process with `--enable-lora`, so the 141 GB
download is paid once rather than per organism.

Pick the offer with the survey tool rather than by eye. It ranks on the cost of
the whole job, because the download is billed time and a cheap box on a slow
link pays for its weights twice over:

```bash
uv run python script/remote/survey_gpu_offers.py --weights-gb 141
```

An earlier version of this file specified 4x H100 SXM on a narrower search. The
wider survey beats it on both cost and memory: at survey time the best offer was
4x H200 at $10.53/hr with 99.7% reliability, 1 TB system RAM and 562 GB VRAM,
against $12.27/hr for the cheapest qualifying 4x H100 SXM. Offers churn, so
re-run the survey rather than trusting the number here.

The survey's generation-hours and job-cost columns are modelled from memory
bandwidth, not measured. Use them to compare offers, never as a runtime estimate.

### Scripts

| Step | Script | Runs where |
|---|---|---|
| provision, sync, launch | `script/remote/r2_fleet.sh` | local |
| per-seat collection | `script/remote/r2_collect_seat.sh` | box |
| capture check before destroy | `script/remote/verify_remote_capture.py` | local |
| gated destroy | `script/remote/gate_and_destroy_boxes.sh` | local |
| publish to HuggingFace | `script/data/publish_dataset.py` | local |

### Order of operations

1. Build the prompt sets locally. Two groups per organism, matched instructions,
   only the affiliation cue swapped. No GPU needed, so do it before renting.
2. Conjecture the behavior axes locally with Haiku. The conjecturer is told only
   what the two user groups are, never what difference to look for.
3. Rent, sync, download weights, serve with vLLM.
4. Collect replies for base and each organism on both groups.
5. **Capture before destroy.** `gate_and_destroy_boxes.sh check` must exit 0.
6. Score locally, compare locally.
7. Publish every artifact to the HuggingFace dataset repo.

### Where the data goes

`unrulyabstractions/apart-idt-data`, private, under a new `auditbench/` prefix:

```bash
uv run python script/data/publish_dataset.py \
    --out-root out/auditbench --path-prefix auditbench --include-all
```

Nothing is considered done until this has run and the file count is logged in
`tmp/VERIFICATION_LOG.md`. A rented box is the only copy of what it produced
until then.

---

## Log

| When | What |
|---|---|
| 2026-08-03 14:45 | Verified adapters ungated. Surveyed 4x H100 offers. Nothing rented. |
| 2026-08-03 16:20 | Tested the base gate instead of inferring it: every file returns 403, so the earlier "our token has access" was wrong and is corrected above. Checked five ungated mirrors, none byte-identical. Re-surveyed offers, 4x H200 beats 4x H100 on cost and memory. Nothing rented, nothing billing. |
