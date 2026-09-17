"""Classify every synapse onto a descending neuron as AXONAL (post site in the VNC) or DENDRITIC (post site in the
brain), from the MaleCNS v1.0 per-synapse partners table. Writes Fly-Lab/results/dn_axon_inputs.feather (the DN slice)
and results/dn_axon_summary.json (per DN type: counts, and the presynaptic types onto DNa01 / DNp01).

WHY (2026-09-17): Ceballos et al., iScience 2026 — axo-axonic inputs onto DNs can veto/amplify/synchronize spikes; our
runtime sums everything at the soma. A DN's dendrite is in the brain and its axon runs down the VNC, so for DNs the
ROI of the postsynaptic site IS the compartment. CHOSEN: 'axonal' = primary_post ROI is a VNC neuromere/nerve
(name contains one of the VNC tokens below); everything else = dendritic. MEASURED: the counts.
Author: Claude Code. Runs on the fly-brain venv (pyarrow + pandas).
"""
from __future__ import annotations
import json, re, sys, time
from pathlib import Path
import pyarrow.feather as pf
import pyarrow.compute as pc
import pyarrow as pa
import pandas as pd

LAB = Path(__file__).resolve().parent
SRC = LAB / "data" / "syn-partners-traced.feather"
ANN = LAB.parent / "AI-Shared" / "projects" / "fly-brain" / "runtime" / "data" / "body-annotations.feather"
OUT = LAB / "results"; OUT.mkdir(exist_ok=True)

# VNC ROI name tokens in MaleCNS (neuromeres, tectulum, nerves). CHOSEN list; anything matching = VNC = axonal for a DN.
VNC_TOKENS = ("LegNp", "WTct", "HTct", "IntTct", "LTct", "NTct", "Ov(", "ANm", "mVAC", "CvN", "ADMN", "PDMN", "DMetaN",
              "MesoAN", "MetaAN", "ProAN", "DProN", "VProN", "ProLN", "MesoLN", "MetaLN", "AbN", "AbNT", "LegNv", "VNC")

t0 = time.perf_counter()
ann = pd.read_feather(ANN)
dn = ann[ann["superclass"].astype(str).str.contains("descending", case=False)]
dn_ids = pa.array(dn["bodyId"].astype("int64").to_numpy())
type_of = dict(zip(dn["bodyId"].astype(int), dn["type"].astype(str)))
type_all = dict(zip(ann["bodyId"].astype(int), ann["type"].astype(str)))
print(f"{len(dn)} descending bodies", flush=True)

# stream the 124M-row table by record batch, keep rows whose post body is a DN
reader = pa.ipc.open_file(pa.memory_map(str(SRC), "r"))
keep = []
for i in range(reader.num_record_batches):
    b = reader.get_batch(i)
    m = pc.is_in(b.column("body_post"), value_set=dn_ids)
    sub = b.filter(m)
    if sub.num_rows:
        keep.append(sub)
    if i % 50 == 0:
        print(f"  batch {i}/{reader.num_record_batches}  kept {sum(k.num_rows for k in keep):,}  {time.perf_counter()-t0:.0f}s", flush=True)
tbl = pa.Table.from_batches(keep) if keep else reader.read_all().slice(0, 0)
df = tbl.to_pandas()
df["primary_post"] = df["primary_post"].astype(str)
df["axonal"] = df["primary_post"].apply(lambda r: any(tok in r for tok in VNC_TOKENS))
df["post_type"] = df["body_post"].map(type_of)
df["pre_type"] = df["body_pre"].map(type_all).fillna("?")
pf.write_feather(pa.Table.from_pandas(df), OUT / "dn_axon_inputs.feather")
print(f"DN input synapses kept: {len(df):,}  axonal: {int(df['axonal'].sum()):,} ({df['axonal'].mean()*100:.1f}%)  {time.perf_counter()-t0:.0f}s", flush=True)

summary = {"source": SRC.name, "rows_kept": int(len(df)), "axonal_rows": int(df["axonal"].sum()),
           "chosen_vnc_tokens": list(VNC_TOKENS), "per_type": {}, "focus": {}}
g = df.groupby("post_type")["axonal"].agg(["size", "sum"])
for t_, r in g.iterrows():
    summary["per_type"][t_] = {"inputs": int(r["size"]), "axonal": int(r["sum"])}
for focus in ("DNa01", "DNp01", "DNa02"):
    sub = df[df["post_type"] == focus]
    ax = sub[sub["axonal"]]
    summary["focus"][focus] = {
        "bodies": sorted(int(x) for x in sub["body_post"].unique()),
        "inputs": int(len(sub)), "axonal": int(len(ax)), "axonal_pct": round(len(ax) / len(sub) * 100, 1) if len(sub) else None,
        "axonal_rois": sub[sub["axonal"]]["primary_post"].value_counts().head(8).to_dict(),
        "axonal_pre_types_top": ax["pre_type"].value_counts().head(15).to_dict(),
        "dendritic_rois_top": sub[~sub["axonal"]]["primary_post"].value_counts().head(5).to_dict(),
    }
(OUT / "dn_axon_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print(json.dumps(summary["focus"], indent=1))
