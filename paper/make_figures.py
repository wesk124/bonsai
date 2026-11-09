#!/usr/bin/env python3
"""Regenerate six PNG-only paper figures from portable, SHA-checked data.

Default: python paper/make_figures.py
Initial trusted snapshot: python paper/make_figures.py --snapshot-from /path/to/bonsai
Dependencies: numpy, matplotlib; seaborn is optional. No benchmark runtime needed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parent
CONTEXTS = (4096, 8192, 16384)
PRECISIONS = ("bf16", "int8", "int4")
BACKENDS = ("pytorch_packed_v1", "triton_packed_v1")
COLORS = {"bf16": "#4FB9AF", "int8": "#6686C5", "int4": "#D59AB5", "text_refill": "#08415C"}
BACKEND_COLORS = ("#4FB9AF", "#6686C5")
SOURCES = {
    "final-campaign.json": "results/analysis-final/campaign/campaign.json",
    "final-analysis.json": "results/analysis-final/analysis.json",
    "pilot-summary.json": "runs/fused-20261108/analysis/summary.json",
    "codec-aggregate.json": "runs/fused-20261108/codec/aggregate.json",
    "accepted-final-manifest.json": "results/analysis-final/artifact_manifest.json",
    "accepted-pilot-manifest.json": "runs/fused-20261108/artifact_manifest.json",
}
CAPTIONS = {
    "final-quality-bytes.png": "Original 900-prompt final evaluation with the PyTorch reference backend. Left: observed target recovery for native BF16, INT8 and INT4 transfer, with 300 unique prompts at each context budget (100 per task). All observed scores are at the ceiling; this does not establish the two-percentage-point preservation margin. Right: mean application handoff bytes, including payload and counted metadata/control frames but excluding network headers. Local encoded sizes are not plotted as transmitted bytes.",
    "final-handoff-latency.png": "Original 900-prompt final evaluation: measured complete native GPU RDMA handoff latency from sender cache-ready to receiver-ready acknowledgement, including codec, allocation, registration and control work. Solid lines show medians and dashed lines show empirical 95th percentiles over 300 unique prompts per context and condition. These percentiles are descriptive, not confidence intervals. Text refill includes receiver tokenization and prefill.",
    "pilot-paired-handoff.png": "Exploratory comparison on nine unique pilot prompts repeated in three process blocks. Thin lines pair the same prompt and block across the reference and fused backends; each context/backend/precision has nine observations from three unique prompts. Large markers show medians and whiskers show empirical 5th–95th percentiles, not confidence intervals. BF16 is optimized in the fused backend too. Blocks 1 and 2 overlapped on disjoint node pairs after block 0 completed; the comparison does not establish a population speedup.",
    "codec-phase-latency.png": "Codec-only encode and decode measurements on nine unique real model caches, with five timed repetitions per cache/backend/precision. Each point is the median across three prompts at that context of each prompt's five-repeat median. Logarithmic latency axes make both reference and fused measurements visible. Codec timings exclude complete network handoff; separately aggregated phase medians need not sum to the combined-latency median.",
    "pilot-handoff-memory.png": "Exploratory 16K pilot handoff memory: median additional peak allocation above the sender's or receiver's measured handoff baseline, for three unique prompts repeated in three blocks. Receiver values include reconstructed BF16 cache residency, not only temporary codec workspace. Per-request registration remains unchanged. Bars are descriptive medians and are not confidence bounds.",
    "pilot-paired-bf16-ratios.png": "Exploratory same-backend comparison: INT8 or INT4 complete handoff divided by BF16 complete handoff for the same prompt and block. Lines show the median of nine paired ratios at each context (three unique prompts repeated in three blocks); vertical ranges span the three block medians, which are also shown as small points. Values below one indicate lower measured latency in this pilot. Ranges are descriptive, not confidence intervals; scheduling and node-placement variation remain possible.",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_tree(value, label):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label}: NaN/Inf count >= 1")
    if isinstance(value, dict):
        for key, item in value.items(): finite_tree(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value): finite_tree(item, f"{label}[{index}]")


def finite(values, label):
    array = np.asarray(values, dtype=float)
    bad = int((~np.isfinite(array)).sum())
    if bad:
        raise ValueError(f"{label}: NaN/Inf count = {bad}")
    return array


def read_json(path):
    value = json.loads(Path(path).read_text())
    finite_tree(value, str(path))
    return value


def manifest_entry(manifest, source, repository):
    candidates = (str(source.resolve()), str(source.relative_to(repository)))
    candidates += (str(source.relative_to(repository / "runs/fused-20261108")),) if source.is_relative_to(repository / "runs/fused-20261108") else ()
    for key in candidates:
        if key in manifest["sha256"]:
            if manifest["sha256"][key] != sha(source):
                raise ValueError(f"accepted source artifact hash mismatch: {source}")
            return key
    raise ValueError(f"source missing from accepted artifact manifest: {source}")


def snapshot(repository, data_dir):
    repository = Path(repository).resolve()
    accepted = {"final": read_json(repository / SOURCES["accepted-final-manifest.json"]),
                "pilot": read_json(repository / SOURCES["accepted-pilot-manifest.json"])}
    if any(m.get("status") != "complete_and_verified" for m in accepted.values()):
        raise ValueError("only accepted complete_and_verified artifact packages may be snapshotted")
    data_dir.mkdir(parents=True, exist_ok=True)
    entries = {}
    for name, relative in SOURCES.items():
        source = repository / relative
        record = {"original_relative_path": relative, "original_sha256": sha(source)}
        if not name.startswith("accepted-"):
            group = "final" if name.startswith("final-") else "pilot"
            record["accepted_manifest"] = f"accepted-{group}-manifest.json"
            record["accepted_manifest_entry"] = manifest_entry(accepted[group], source, repository)
        (data_dir / name).write_bytes(source.read_bytes())
        record["packaged_sha256"] = sha(data_dir / name)
        entries[name] = record
    pilot = read_json(data_dir / "pilot-summary.json")
    measurements, raw_sources = {}, []
    for source in pilot["inputs"]:
        original = Path(source["path"])
        # Resolve recorded absolute paths through the requested repository when
        # a checkout has moved, retaining the original logical relative path.
        logical = Path("runs/fused-20261108/comparison") / original.name
        path = repository / logical
        if sha(path) != source["sha256"]:
            raise ValueError(f"raw paired input does not match accepted summary: {path}")
        raw_sources.append({"original_relative_path": str(logical), "sha256": source["sha256"]})
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip(): continue
            row = json.loads(line)
            finite_tree(row, f"{path}:{number}")
            if row["condition"] != "transfer": continue
            if row["kind"] != "prediction" or row["codec_backend"] != source["backend"] or row["transport"] != "fabric-lib":
                raise ValueError("invalid accepted paired transfer measurement")
            key = (source["block"], row["sample_id"], row["context_tokens"], row["precision"])
            cell = measurements.setdefault(key, {"sample_hash": row["sample_hash"], "backend_ms": {}, "generated_ids": row["generated_ids"]})
            if cell["sample_hash"] != row["sample_hash"] or cell["generated_ids"] != row["generated_ids"] or source["backend"] in cell["backend_ms"]:
                raise ValueError("paired snapshot lineage/token mismatch or duplicate")
            cell["backend_ms"][source["backend"]] = row["handoff_ack_ms"]
    records = [{"block": b, "sample_id": sample, "context_tokens": context, "precision": precision,
                "sample_hash": values["sample_hash"], "backend_ms": values["backend_ms"]}
               for (b, sample, context, precision), values in sorted(measurements.items())]
    paired = {"scope": "81 exact same-block/same-prompt transfer pairs; nine unique pilot prompts repeated in three blocks",
              "source_summary_sha256": entries["pilot-summary.json"]["original_sha256"], "raw_source_sha256": raw_sources, "records": records}
    pair_path = data_dir / "pilot-handoff-pairs.json"
    pair_path.write_text(json.dumps(paired, indent=2, allow_nan=False)+"\n")
    entries[pair_path.name] = {"derived_from": "pilot-summary.json and its SHA-verified original comparison inputs",
                               "original_relative_path": None, "original_sha256": None, "packaged_sha256": sha(pair_path)}
    package = {"schema_version": 1, "datasets": entries,
               "scope": "Exact accepted JSON snapshots plus compact paired timings; no full raw benchmark outputs are required for figure regeneration."}
    (data_dir / "manifest.json").write_text(json.dumps(package, indent=2)+"\n")


def load_data(data_dir):
    package = read_json(data_dir / "manifest.json")
    result = {}
    for name, record in package["datasets"].items():
        if Path(name).name != name:
            raise ValueError("unsafe snapshot filename")
        path = data_dir / name
        if sha(path) != record["packaged_sha256"]:
            raise ValueError(f"portable input SHA256 mismatch: {name}")
        if record.get("original_sha256") and record["original_sha256"] != record["packaged_sha256"]:
            raise ValueError(f"snapshot differs from original artifact: {name}")
        result[name] = read_json(path)
    for name, record in package["datasets"].items():
        if record.get("accepted_manifest"):
            accepted = result[record["accepted_manifest"]]
            if accepted.get("status") != "complete_and_verified" or accepted["sha256"][record["accepted_manifest_entry"]] != record["original_sha256"]:
                raise ValueError(f"accepted lineage verification failed: {name}")
    validate_data(result, package)
    return result, package


def unique_map(rows, keys, label):
    output = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in output: raise ValueError(f"{label}: duplicate {key}")
        output[key] = row
    return output


def validate_data(data, package):
    campaign = data["final-campaign.json"]
    analysis = data["final-analysis.json"]
    pilot = data["pilot-summary.json"]
    codec = data["codec-aggregate.json"]
    pairs = data["pilot-handoff-pairs.json"]
    if not campaign["final_matrix_complete"] or not analysis["complete_final_verified"]:
        raise ValueError("incomplete original final evaluation")
    if len(campaign["campaigns"]) != 1 or len(campaign["campaigns"][0]["strata"]) != 1:
        raise ValueError("expected one compatible original final campaign")
    final = campaign["campaigns"][0]["strata"][0]
    if (final["split"], final["unique_prompts"], final["total_rows"], final["execution_failures"]) != ("final",900,6300,0):
        raise ValueError("original final counts do not match the accepted matrix")
    final_cells = unique_map(final["by_context"], ("context_tokens","condition","precision"), "final")
    for context in CONTEXTS:
        for condition, precision in [("transfer",p) for p in PRECISIONS]+[("text_refill","bf16")]:
            row = final_cells[(context,condition,precision)]
            if row["predictions"] != 300 or row["attempted_prompts"] != 300 or row["transport"] != "fabric-lib":
                raise ValueError("final transfer cell count/scope mismatch")
            if sorted(row["task_counts"].values()) != [100,100,100]:
                raise ValueError("final task counts mismatch")
            finite([row["target_recovery"]["mean"],row["wire_mib"]["mean"],row["handoff_ms"]["p50"],row["handoff_ms"]["p95"]], f"final plotted {context}/{condition}/{precision}")
    if (pilot["status"],pilot["pilot_unique_prompts"],pilot["blocks"],pilot["predictions"],pilot["cross_backend_exact_token_pairs"]) != ("PASS",9,3,378,189):
        raise ValueError("pilot acceptance/counts mismatch")
    if len(pilot["inputs"]) != 6 or any(r["handoff_equivalence"]["exact_token_id_pairs"] != 36 or r["handoff_equivalence"]["native_handoff_equivalence"] != "PASS" for r in pilot["inputs"]):
        raise ValueError("pilot within-backend handoff equivalence incomplete")
    if (codec["status"],len(codec["cells"])) != ("PASS",18):
        raise ValueError("codec acceptance/counts mismatch")
    codec_cells = unique_map(codec["cells"],("context_tokens","precision","backend"),"codec")
    for context in CONTEXTS:
        for precision in PRECISIONS:
            for backend in BACKENDS:
                row = codec_cells[(context,precision,backend)]
                if row["unique_prompts"] != 3 or min(finite([row["encode_ms"],row["decode_ms"]], f"codec {context}/{precision}/{backend}")) <= 0:
                    raise ValueError("invalid codec plotted measurement")
    if pairs["source_summary_sha256"] != package["datasets"]["pilot-summary.json"]["packaged_sha256"]:
        raise ValueError("paired snapshot is not bound to the accepted pilot summary")
    expected_raw = {Path(r["path"]).name:r["sha256"] for r in pilot["inputs"]}
    if {Path(r["original_relative_path"]).name:r["sha256"] for r in pairs["raw_source_sha256"]} != expected_raw:
        raise ValueError("paired raw input hashes disagree with accepted pilot summary")
    records = pairs["records"]
    unique_map(records,("block","sample_id","context_tokens","precision"),"paired data")
    if len(records) != 81 or {r["block"] for r in records} != {0,1,2} or len({r["sample_id"] for r in records}) != 9:
        raise ValueError("paired snapshot requires 81 pairs over nine unique prompts and three blocks")
    pilot_cells = unique_map(pilot["by_context"],("backend","context_tokens","condition","precision"),"pilot")
    for context in CONTEXTS:
        for precision in PRECISIONS:
            selected = [r for r in records if r["context_tokens"] == context and r["precision"] == precision]
            if len(selected) != 9 or len({r["sample_id"] for r in selected}) != 3 or Counter(r["block"] for r in selected) != {0:3,1:3,2:3}:
                raise ValueError("paired context/precision counts mismatch")
            for backend in BACKENDS:
                values = finite([r["backend_ms"][backend] for r in selected],f"paired latency {context}/{precision}/{backend}")
                if min(values) <= 0: raise ValueError("handoff latency must be positive")
                cell = pilot_cells[(backend,context,"transfer",precision)]
                if (cell["attempted_prompts"],cell["n_unique_prompts"],cell["n_blocks"]) != (9,3,3):
                    raise ValueError("pilot summary observational units mismatch")
                if not np.allclose(np.quantile(values,[.5,.95]),[cell["handoff_ms"]["p50"],cell["handoff_ms"]["p95"]],rtol=1e-12,atol=1e-12):
                    raise ValueError("paired raw data do not reproduce accepted summary percentiles")
                finite([cell["sender_handoff_memory"]["temporary_peak_allocated_mib"]["p50"],cell["receiver_handoff_memory"]["temporary_peak_allocated_mib"]["p50"]],f"pilot plotted memory {context}/{precision}/{backend}")


def render(data, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    try:
        import seaborn as sns
    except ImportError:
        sns = None
        plt.style.use("seaborn-v0_8-white")
        plt.rcParams.update({"xtick.bottom":True,"ytick.left":True})
    else:
        sns.set_style("white",{"xtick.bottom":True,"ytick.left":True})
    plt.rcParams.update({"font.size":8,"axes.titlesize":9,"axes.labelsize":8,"legend.fontsize":7,"xtick.labelsize":7,"ytick.labelsize":7,"figure.titlesize":10,"font.family":"DejaVu Sans"})
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    def finish(fig,name,rect=(0,.16,1,.94)):
        if sns is not None: sns.despine(fig=fig)
        else:
            for ax in fig.axes: ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout(rect=rect)
        path=output/name
        fig.savefig(path,dpi=300,bbox_inches="tight",facecolor="white")
        plt.close(fig)
        paths.append(path)
    def context_axis():
        plt.xticks([4,8,16],["4K","8K","16K"])
        plt.xlabel("Context budget")
    def title(fig,text,note):
        fig.suptitle(text,y=.99)
        fig.text(.5,.045,note,ha="center",va="center",fontsize=6.5)
    final = data["final-campaign.json"]["campaigns"][0]["strata"][0]
    final_cells=unique_map(final["by_context"],("context_tokens","condition","precision"),"final")
    pilot_cells=unique_map(data["pilot-summary.json"]["by_context"],("backend","context_tokens","condition","precision"),"pilot")
    codec_cells=unique_map(data["codec-aggregate.json"]["cells"],("context_tokens","precision","backend"),"codec")
    pairs=data["pilot-handoff-pairs.json"]["records"]
    precision_handles=[Line2D([],[],color=COLORS[p],marker={"bf16":"s","int8":"x","int4":"o"}[p],label=p.upper()) for p in PRECISIONS]
    backend_handles=[Line2D([],[],color=c,marker=m,label=l) for c,m,l in zip(BACKEND_COLORS,("s","D"),("PyTorch reference","Fused Triton"))]

    fig,axs=plt.subplots(1,2,figsize=(6.8,2.75));axes=iter(axs)
    plt.sca(next(axes))
    for p in PRECISIONS:
        values=[100*final_cells[(c,"transfer",p)]["target_recovery"]["mean"] for c in CONTEXTS]
        plt.plot([4,8,16],values,color=COLORS[p],marker={"bf16":"s","int8":"x","int4":"o"}[p],markersize=7 if p=="bf16" else 4.5,markerfacecolor="none",linewidth=1)
    plt.ylim(0,105);plt.ylabel("Target recovery (%)");context_axis();plt.title("(a) Observed retrieval")
    plt.sca(next(axes))
    for p in PRECISIONS:
        plt.plot([4,8,16],[final_cells[(c,"transfer",p)]["wire_mib"]["mean"] for c in CONTEXTS],linestyle="-",marker={"bf16":"s","int8":"x","int4":"o"}[p],color=COLORS[p],markersize=4,linewidth=1)
    plt.ylabel("Mean application handoff (MiB)");context_axis();plt.title("(b) Measured handoff bytes")
    fig.legend(handles=precision_handles,loc="upper center",bbox_to_anchor=(.5,.9),ncol=3,frameon=False)
    title(fig,"Original final evaluation: quality and application bytes","900 unique prompts; 300 per context. Ceiling observations do not establish the 2 pp preservation margin.")
    finish(fig,"final-quality-bytes.png",(0,.15,1,.94))

    fig=plt.figure(figsize=(6.8,2.95))
    handles=[]
    for condition,p,label in [("transfer","bf16","BF16"),("transfer","int8","INT8"),("transfer","int4","INT4"),("text_refill","bf16","Text refill")]:
        color=COLORS["text_refill" if condition=="text_refill" else p]
        for quantile,style,marker in (("p50","-","o"),("p95","--","^")):
            line,=plt.plot([4,8,16],[final_cells[(c,condition,p)]["handoff_ms"][quantile] for c in CONTEXTS],linestyle=style,marker=marker,color=color,markersize=3.5,linewidth=1,label=f"{label} {quantile}")
            handles.append(line)
    context_axis();plt.ylabel("Complete handoff latency (ms)");plt.ylim(bottom=0)
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.91),ncol=4,frameon=False)
    title(fig,"Original final evaluation: complete native GPU RDMA handoff","300 unique prompts/context/condition. p95 is an empirical percentile, not a confidence interval.")
    finish(fig,"final-handoff-latency.png",(0,.14,1,.87))

    fig,axs=plt.subplots(1,3,figsize=(6.8,2.95),sharey=True);axes=iter(axs)
    maximum=max(v for r in pairs for v in r["backend_ms"].values())
    for context in CONTEXTS:
        plt.sca(next(axes))
        for x,p in enumerate(PRECISIONS):
            selected=[r for r in pairs if r["context_tokens"]==context and r["precision"]==p]
            values=np.array([[r["backend_ms"][b] for b in BACKENDS] for r in selected])
            for before,after in values: plt.plot([x-.13,x+.13],[before,after],color="#9596C6",alpha=.3,linewidth=.55,zorder=1)
            for b in range(2):
                position=x+(-.13 if b==0 else .13);low,median,high=np.quantile(values[:,b],[.05,.5,.95])
                plt.scatter(np.full(9,position),values[:,b],color=BACKEND_COLORS[b],s=7,alpha=.4,zorder=2)
                plt.errorbar(position,median,yerr=[[median-low],[high-median]],color=BACKEND_COLORS[b],marker=("s","D")[b],markersize=4,capsize=2,linewidth=1,zorder=3)
        plt.xticks(range(3),[p.upper() for p in PRECISIONS]);plt.xlim(-.5,2.5);plt.ylim(0,maximum*1.08)
        plt.xlabel("Transfer precision");plt.title(f"{context//1024}K context budget")
        if context==4096:plt.ylabel("Complete handoff (ms)")
    fig.legend(handles=backend_handles,loc="upper center",bbox_to_anchor=(.5,.9),ncol=2,frameon=False)
    title(fig,"Exploratory pilot: paired complete handoff","3 unique prompts × 3 blocks/context/backend; medians and empirical 5th–95th percentiles, not confidence intervals.")
    finish(fig,"pilot-paired-handoff.png",(0,.15,1,.94))

    fig,axs=plt.subplots(1,2,figsize=(6.8,2.9));axes=iter(axs)
    for phase,panel in (("encode_ms","(a) Encoding"),("decode_ms","(b) Reconstruction")):
        plt.sca(next(axes))
        for p in PRECISIONS:
            for b,backend in enumerate(BACKENDS):
                plt.plot([4,8,16],[codec_cells[(c,p,backend)][phase] for c in CONTEXTS],linestyle=("-","--")[b],marker=("o","D")[b],color=COLORS[p],markersize=3.5,linewidth=1)
        plt.yscale("log");plt.ylim(1,100)
        plt.yticks([1,2,5,10,20,50,100],["1","2","5","10","20","50","100"])
        from matplotlib.ticker import NullFormatter
        plt.gca().yaxis.set_minor_formatter(NullFormatter())
        context_axis();plt.ylabel("Codec latency (ms; log scale)");plt.title(panel)
    style_handles=[Line2D([],[],color="#08415C",linestyle=s,marker=m,label=l) for s,m,l in (("-","o","Reference"),("--","D","Fused"))]
    codec_precision_handles=[Line2D([],[],color=COLORS[p],label=p.upper()) for p in PRECISIONS]
    fig.legend(handles=codec_precision_handles+style_handles,loc="upper center",bbox_to_anchor=(.5,.91),ncol=5,frameon=False)
    title(fig,"Exploratory codec-only measurements","Median over 3 unique caches/context of each cache's 5-repeat median; excludes complete network handoff.")
    finish(fig,"codec-phase-latency.png",(0,.15,1,.94))

    fig,axs=plt.subplots(1,2,figsize=(6.8,2.8),sharey=True);axes=iter(axs)
    for location,panel in (("sender_handoff_memory","(a) Sender"),("receiver_handoff_memory","(b) Receiver")):
        plt.sca(next(axes))
        for b,backend in enumerate(BACKENDS):
            values=[pilot_cells[(backend,16384,"transfer",p)][location]["temporary_peak_allocated_mib"]["p50"]/1024 for p in PRECISIONS]
            positions=np.arange(3)+(-.18 if b==0 else .18)
            plt.bar(positions,values,width=.32,color=BACKEND_COLORS[b])
            for x,y in zip(positions,values):plt.text(x,y+.07,f"{y:.2f}",ha="center",va="bottom",fontsize=6.5)
        plt.xticks(range(3),[p.upper() for p in PRECISIONS]);plt.xlabel("Transfer precision");plt.title(panel);plt.ylim(0,7.6)
        if location=="sender_handoff_memory":plt.ylabel("Additional peak allocation (GiB)")
    fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(BACKEND_COLORS,("PyTorch reference","Fused Triton"))],loc="upper center",bbox_to_anchor=(.5,.9),ncol=2,frameon=False)
    title(fig,"Exploratory 16K pilot: handoff memory","Medians over 3 unique prompts × 3 blocks. Receiver includes reconstructed cache residency; registration unchanged.")
    finish(fig,"pilot-handoff-memory.png",(0,.15,1,.94))

    pair_lookup={(r["block"],r["sample_id"],r["precision"]):r for r in pairs}
    fig,axs=plt.subplots(1,2,figsize=(6.8,2.95),sharey=True);axes=iter(axs)
    ratio_maximum=max(r["backend_ms"][backend]/pair_lookup[(r["block"],r["sample_id"],"bf16")]["backend_ms"][backend]
                      for r in pairs if r["precision"] in ("int8","int4") for backend in BACKENDS)
    for p in ("int8","int4"):
        plt.sca(next(axes));plt.axhline(1,color="#08415C",linestyle=":",linewidth=.9)
        for b,backend in enumerate(BACKENDS):
            medians=[]
            for x,context in zip([4,8,16],CONTEXTS):
                selected=[r for r in pairs if r["context_tokens"]==context and r["precision"]==p]
                ratios=[r["backend_ms"][backend]/pair_lookup[(r["block"],r["sample_id"],"bf16")]["backend_ms"][backend] for r in selected]
                medians.append(float(np.median(ratios)))
                block_medians=[float(np.median([ratio for ratio,r in zip(ratios,selected) if r["block"]==block])) for block in range(3)]
                plt.vlines(x,min(block_medians),max(block_medians),color=BACKEND_COLORS[b],linewidth=1.4)
                plt.scatter([x]*3,block_medians,color=BACKEND_COLORS[b],marker=("o","D")[b],s=10,alpha=.6)
            plt.plot([4,8,16],medians,color=BACKEND_COLORS[b],marker=("s","D")[b],markersize=4,linewidth=1)
        context_axis();plt.title(p.upper());plt.ylim(0,ratio_maximum*1.08)
        if p=="int8":plt.ylabel("Paired handoff ratio to same-backend BF16")
    fig.legend(handles=backend_handles,loc="upper center",bbox_to_anchor=(.5,.9),ncol=2,frameon=False)
    title(fig,"Exploratory pilot: quantized handoff versus each backend's BF16","Lines: median of 9 paired repeat ratios/context. Small points and ranges: 3 block medians, not confidence intervals.")
    finish(fig,"pilot-paired-bf16-ratios.png",(0,.15,1,.94))
    return paths


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-from",type=Path,help="explicitly refresh portable snapshots from accepted repository artifacts")
    parser.add_argument("--data",type=Path,default=BASE/"data")
    parser.add_argument("--out",type=Path,default=BASE/"figures")
    args=parser.parse_args(argv)
    # This authoring tool owns only paper/data and paper/figures.
    if args.data.resolve()!=BASE/"data" or args.out.resolve()!=BASE/"figures":
        parser.error("data/output directories must be this portable paper's data/ and figures/")
    if args.snapshot_from:snapshot(args.snapshot_from,args.data)
    data,package=load_data(args.data)
    paths=render(data,args.out)
    manifest={"schema_version":1,"reproduce":"python paper/make_figures.py","dpi":300,"format":"PNG only",
              "data_manifest_sha256":sha(args.data/"manifest.json"),"script_sha256":sha(__file__),
              "input_sha256":{name:record["packaged_sha256"] for name,record in package["datasets"].items()},
              "figures":{p.name:{"sha256":sha(p),"caption":CAPTIONS[p.name]} for p in paths}}
    (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"figures":len(paths),"out":str(args.out),"data_manifest_sha256":manifest["data_manifest_sha256"]}))


if __name__=="__main__":main()
