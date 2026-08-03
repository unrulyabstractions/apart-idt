# Cloud jobs

One row per rented machine, and everything a second person needs to take it over.
Update the status line **every time** you start, check, or stop a job. A stale
status here is worse than none: the next person will trust it.

**Last status:** 2026-08-03 14:45 UTC-0700
**Written by:** assistant session `a1f6544a`
**Running now:** nothing. No vast.ai instance is rented.

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

Credentials live in the shell environment, never on the box:
`HF_TOKEN` is needed on the box to pull the gated base model.
`OPENAI_API_KEY` and `ANTHROPIC_API_KEY` stay local; scoring runs locally so no
API key ever travels to a rented machine.

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
- Base `meta-llama/Llama-3.3-70B-Instruct` is `gated=manual` and **our token has
  access**. 141 GB in bf16 safetensors.

### Machine

**4x H100 SXM 80GB**, tensor parallel 4.

Why: the base is 141 GB in bf16 and behavioral auditing must not quantize, since
quantization changes the very outputs being measured. Two H100s (160 GB) leaves
under 20 GB for KV cache and activations across the pair, which throttles batch
size and risks OOM mid-run. Four gives 320 GB, so the weights sit in half the
memory and the rest is headroom. vLLM serves the base and all three adapters from
one process with `--enable-lora`, so the 141 GB is paid once rather than per
organism.

Require when searching offers:

```bash
vastai search offers 'num_gpus=4 gpu_name=H100_SXM reliability>0.98 \
  disk_space>400 inet_down>500' -o 'dph'
```

- `disk_space>400`: 141 GB base + 3 adapters + outputs, with room to spare.
- `inet_down>500`: the base download dominates setup. At 640 Mbps it is roughly
  30 minutes; a slow box can spend hours before the first token.
- `reliability>0.98`: a mid-run eviction costs the whole download again.

At survey time the best value was ~$6.93/hr at 99.8% reliability, 408 GB disk,
515 GB RAM. Budget the download as billed time.

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
| 2026-08-03 14:45 | Verified adapters ungated and base accessible. Surveyed 4x H100 offers. Nothing rented. |
