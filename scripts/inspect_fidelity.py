import json
from pathlib import Path
import numpy as np

ROOT = Path("/home/nacho/projects/fault_converters")
EXPERIMENT = ROOT / "experiments" / "dataset_01_run_06_corrected"
MODELS = ["conv1d", "gru", "lstm", "mlp", "transformer"]

for model in MODELS:
    print(f"=== {model.upper()} ===")
    dep_p = EXPERIMENT / f"carla_{model}" / "deployment"
    rep_p = dep_p / "unified_deployment_report.json"
    cons_p = dep_p / "cross_backend_consistency.json"
    rep = json.loads(rep_p.read_text()) if rep_p.exists() else {}
    cons = {r["model_name"]: r for r in json.loads(cons_p.read_text())} if cons_p.exists() else {}
    
    for k, val in rep.items():
        ev = val.get("evaluation", {})
        f1 = ev.get("f1")
        acc = ev.get("accuracy")
        dis = ev.get("anomaly_disagreement_rate_vs_baseline", cons.get(k, {}).get("anomaly_disagreement_rate_vs_baseline", 0.0))
        lat = val.get("performance", {}).get("latency_mean_ms")
        print(f"  {k:22s} | Lat: {lat:7.2f} ms | Acc: {acc} | F1: {f1} | Disagree: {dis*100:5.2f}%")
    
    for prec in ("fp32", "fp16"):
        np_p = dep_p / f"native_gpu_{prec}.json"
        if np_p.exists():
            n = json.loads(np_p.read_text())
            ev = n.get("evaluation", {})
            lat = ev.get("latency_per_sample_mean_ms")
            acc = ev.get("accuracy")
            f1 = ev.get("f1")
            dis = ev.get("full_test_anomaly_disagreement_rate_vs_baseline", 0.0)
            print(f"  {'Native GPU ' + prec.upper():22s} | Lat: {lat:7.2f} ms | Acc: {acc} | F1: {f1} | Disagree: {dis*100:5.2f}%")
