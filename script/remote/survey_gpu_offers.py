"""Rank vast.ai offers by the cost of a whole job, not by the hourly rate.

    uv run python script/remote/survey_gpu_offers.py --weights-gb 141

A rented box bills from the moment it starts, so the weight download is paid
time. A cheap box on a slow link can pay for its own weights twice over, and a
cheap box with slow memory pays again on every token. Both are priced here.

Generation is memory-bandwidth bound, so aggregate HBM bandwidth predicts
throughput far better than FLOPs. Cards wired only over PCIe pay a further
tensor-parallel penalty on a model this size, which is applied below.

The generation-hours column is a MODEL, not a measurement. It scales a nominal
work constant by bandwidth so that offers can be compared against each other.
Do not quote it as a runtime.
"""

from __future__ import annotations

import argparse
import json
import subprocess

#: Aggregate HBM bandwidth per GPU, TB/s.
BANDWIDTH_TB_S = {
    "H200": 4.8, "H100_SXM": 3.35, "H100_NVL": 3.9, "H100_PCIE": 2.0,
    "A100_SXM4": 2.04, "A100_PCIE": 1.94, "L40S": 0.864,
    "RTX 6000Ada": 0.96, "RTX A6000": 0.768, "RTX 4090": 1.008,
    "B200": 8.0, "H20": 4.0,
}

#: Cards with a real interconnect. Anything else sharding a 70B over more than
#: two GPUs spends its time in allreduce.
NVLINK = {"H200", "H100_SXM", "H100_NVL", "A100_SXM4", "B200"}
PCIE_TP_PENALTY = 0.65

#: Nominal generation work, in TB/s-hours. Only its ratio between offers matters.
NOMINAL_WORK = 40.0

QUERIES = (
    "num_gpus=2 gpu_name=H200", "num_gpus=4 gpu_name=H200",
    "num_gpus=4 gpu_name=H100_SXM", "num_gpus=8 gpu_name=H100_SXM",
    "num_gpus=4 gpu_name=H100_NVL", "num_gpus=4 gpu_name=H100_PCIE",
    "num_gpus=4 gpu_name=A100_SXM4", "num_gpus=8 gpu_name=A100_SXM4",
    "num_gpus=8 gpu_name=A100_PCIE", "num_gpus=4 gpu_name=L40S",
    "num_gpus=8 gpu_name=L40S", "num_gpus=4 gpu_name=RTX_6000Ada",
    "num_gpus=8 gpu_name=RTX_A6000",
)


def survey(weights_gb: float, filters: str, headroom: float) -> list[dict]:
    rows: list[dict] = []
    seen: set[int] = set()
    for query in QUERIES:
        proc = subprocess.run(
            ["vastai", "search", "offers", f"{query} {filters}", "-o", "dph", "--raw"],
            capture_output=True, text=True)
        try:
            offers = json.loads(proc.stdout)
        except json.JSONDecodeError:
            continue
        for offer in offers[:6]:
            if offer["id"] in seen:
                continue
            total_vram = offer["num_gpus"] * offer["gpu_ram"] / 1024
            # A bare fit leaves nothing for the KV cache and will OOM mid-run.
            if total_vram < weights_gb * headroom:
                continue
            seen.add(offer["id"])
            name = offer["gpu_name"]
            band = BANDWIDTH_TB_S.get(name, 1.0) * offer["num_gpus"]
            if name not in NVLINK and offer["num_gpus"] > 2:
                band *= PCIE_TP_PENALTY
            down_mbps = offer.get("inet_down") or 1.0
            download_hr = (weights_gb * 8 * 1024) / down_mbps / 3600
            generate_hr = NOMINAL_WORK / band
            dph = offer["dph_total"]
            rows.append({
                "id": offer["id"],
                "gpus": f"{offer['num_gpus']}x{name}",
                "vram": total_vram,
                "spare": total_vram - weights_gb,
                "dph": dph,
                "band": band,
                "download_hr": download_hr,
                "generate_hr": generate_hr,
                # +0.5h for provisioning, install, and load.
                "job_cost": (download_hr + generate_hr + 0.5) * dph,
                "reliability": offer["reliability2"] * 100,
                "disk": offer["disk_space"],
                "cpu_ram": (offer.get("cpu_ram") or 0) / 1024,
                "geo": (offer.get("geolocation") or "?").strip(),
            })
    rows.sort(key=lambda r: r["job_cost"])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights-gb", type=float, default=141.0,
                    help="model size in bf16; 141 for Llama-3.3-70B")
    ap.add_argument("--headroom", type=float, default=1.15,
                    help="required VRAM as a multiple of the weights")
    ap.add_argument("--filters", default="reliability>0.97 disk_space>350 "
                                         "inet_down>300 rentable=true")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    rows = survey(args.weights_gb, args.filters, args.headroom)
    if not rows:
        print("no offer can hold this model with that headroom")
        return
    print(f"{'id':>10} {'gpus':<16} {'vram':>6} {'spare':>6} {'$/hr':>7} {'BW':>7} "
          f"{'dl':>5} {'gen*':>5} {'JOB$*':>7} {'rel':>5} {'ram':>6}  geo")
    for r in rows[:args.top]:
        print(f"{r['id']:>10} {r['gpus']:<16} {r['vram']:>5.0f}G {r['spare']:>5.0f}G "
              f"{r['dph']:>7.2f} {r['band']:>6.1f}T {r['download_hr']:>4.1f}h "
              f"{r['generate_hr']:>4.1f}h {r['job_cost']:>7.2f} "
              f"{r['reliability']:>5.1f} {r['cpu_ram']:>5.0f}G  {r['geo']}")
    print(f"\n{len(rows)} offers fit {args.weights_gb:.0f}GB at {args.headroom:.2f}x headroom.")
    print("* gen and JOB$ are modelled from bandwidth, not measured. Ratios only.")


if __name__ == "__main__":
    main()
