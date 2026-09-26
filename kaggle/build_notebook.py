"""Build a self-contained Kaggle notebook (code embedded as a base64 tarball).

Variants:
  submit  - competition rerun: serve Qwen with vLLM, play the gateway's hidden games.
            Outside the rerun it writes the placeholder submission Kaggle requires
            and runs a CPU self-check on two public games.
  dev     - play all public games offline with the LLM; outputs runs/ for analysis.
  evolve  - run/resume the offline evolver; state is written to /kaggle/working/evolve.

Usage: python kaggle/build_notebook.py --variant submit [--out build/submit]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["arc_harness", "evolve", "prompts", "skills", "pyproject.toml", "README.md"]


def code_blob() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in INCLUDE:
            p = ROOT / name
            if p.exists():
                tar.add(p, arcname=name, filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return base64.b64encode(buf.getvalue()).decode()


def cell(src: str, kind: str = "code") -> dict:
    c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


SETUP = r'''
import base64, io, os, sys, tarfile, subprocess, glob, json, time
CFG = json.loads({cfg!r})
WORK = "/kaggle/working/arc26"
os.makedirs(WORK, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE)), mode="r:gz").extractall(WORK)
COMP = "/kaggle/input/competitions/" + CFG["competition"]
if not os.path.isdir(COMP):
    COMP = "/kaggle/input/" + CFG["competition"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links",
                COMP + "/arc_agi_3_wheels", "arc-agi", "python-dotenv"], check=False)
os.environ["PYTHONPATH"] = WORK
os.environ["ENVIRONMENTS_DIR"] = COMP + "/environment_files"
sys.path.insert(0, WORK)
RERUN = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
print("rerun:", RERUN, "| comp:", COMP, "| inputs:", sorted(glob.glob("/kaggle/input/*")))
'''

VLLM = r'''
def _find_model_dir():
    for top in sorted(glob.glob("/kaggle/input/*")) + sorted(glob.glob("/kaggle/input/*/*")):
        for root, _, files in os.walk(top):
            if "config.json" in files and any(f.endswith(".safetensors") for f in files):
                return root
    return None

def _find_wheelhouse():
    for root, _, files in os.walk("/kaggle/input"):
        if any(f.startswith("vllm-") and f.endswith(".whl") for f in files):
            return root
    return None

VLLM_PROC = None
def start_vllm():
    global VLLM_PROC
    model_dir, wh = _find_model_dir(), _find_wheelhouse()
    print("model:", model_dir, "| wheelhouse:", wh)
    if not model_dir:
        print("no model mounted -> the agent will run as the pure explorer")
        return False
    try:
        import vllm  # noqa: F401
    except Exception:
        if wh:
            r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", wh, "vllm"],
                               capture_output=True, text=True)
            print("pip vllm:", r.returncode, r.stdout[-500:], r.stderr[-1500:])
    v = CFG["vllm"]
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model_dir,
           "--served-model-name", v["served_model_name"], "--port", "8000",
           "--max-model-len", str(v["max_model_len"]), "--gpu-memory-utilization", str(v["gpu_memory_utilization"]),
           "--max-num-seqs", str(v["max_num_seqs"]), *v["extra_args"]]
    log = open("/kaggle/working/vllm.log", "w")
    VLLM_PROC = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    os.environ["ARC26_LLM_BASE_URL"] = "http://127.0.0.1:8000/v1"
    os.environ["ARC26_LLM_MODEL"] = v["served_model_name"]
    print("vLLM starting:", " ".join(cmd))
    return True
'''

SUBMIT_RUN = r'''
if RERUN:
    have_llm = start_vllm()
    subprocess.run("curl -s --fail --retry 999 --retry-all-errors --retry-delay 5 --retry-max-time 900 "
                   "http://gateway:8001/api/games > /dev/null", shell=True)
    env = {**os.environ, "ARC_BASE_URL": "http://gateway:8001", "ARC_API_KEY": "test-key-123",
           "OPERATION_MODE": "online", "ARC26_LLM_WAIT": "1500"}
    cmd = [sys.executable, "-m", "arc_harness.run", "--mode", "competition",
           "--llm", "vllm" if have_llm else "none", "--hours", str(CFG["hours"]),
           "--concurrency", str(CFG["concurrency"]), "--tag", "submit", "--out", "/kaggle/working/run"]
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=WORK, env=env)
else:
    import pandas as pd
    pd.DataFrame([["1_0", "1", True, 1]], columns=["row_id", "game_id", "end_of_game", "score"]).to_parquet(
        "/kaggle/working/submission.parquet", index=False)
    r = subprocess.run([sys.executable, "-m", "arc_harness.run", "--mode", "offline", "--llm", "none",
                        "--games", "ls20,vc33", "--max-actions", "200", "--hours", "0.05",
                        "--out", "/kaggle/working/selfcheck"], cwd=WORK, capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-2000:])
'''

DEV_RUN = r'''
have_llm = start_vllm()
cmd = [sys.executable, "-m", "arc_harness.run", "--mode", "offline", "--llm", "vllm" if have_llm else "none",
       "--hours", os.getenv("ARC26_DEV_HOURS", "3"), "--concurrency", str(CFG["concurrency"]),
       "--games", os.getenv("ARC26_DEV_GAMES", "all"), "--tag", "dev", "--out", "/kaggle/working/dev_run"]
env = {**os.environ, "ARC26_LLM_WAIT": "1500"}
print(" ".join(cmd))
subprocess.run(cmd, cwd=WORK, env=env)
import pandas as pd
pd.DataFrame([["1_0", "1", True, 1]], columns=["row_id", "game_id", "end_of_game", "score"]).to_parquet(
    "/kaggle/working/submission.parquet", index=False)
if os.path.exists("/kaggle/working/dev_run/summary.json"):
    print(open("/kaggle/working/dev_run/summary.json").read()[:3000])
'''

EVOLVE_RUN = r'''
have_llm = start_vllm()
import shutil, urllib.request
state_dir = "/kaggle/working/evolve"
prev = sorted(glob.glob("/kaggle/input/*/evolve/run_meta.json"))
if prev and not os.path.exists(state_dir):
    shutil.copytree(os.path.dirname(prev[-1]), state_dir)
    print("resuming from", prev[-1])
os.makedirs("/kaggle/working/arc26/data", exist_ok=True)
if not os.path.exists(WORK + "/data/environment_files"):
    os.symlink(COMP + "/environment_files", WORK + "/data/environment_files")
for _ in range(300):
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5); break
    except Exception:
        time.sleep(5)
cfg_path = state_dir + "/evolve_run.json"
if not os.path.exists(cfg_path):
    subprocess.run([sys.executable, "-m", "evolve", "init", "--config", cfg_path, "--work-dir", state_dir,
                    "--repo", WORK], cwd=WORK, check=True)
env = {**os.environ, "ARC26_LLM_WAIT": "600", "ARC26_EVOLVE_HOURS": os.getenv("ARC26_EVOLVE_HOURS", "10")}
subprocess.run([sys.executable, "-m", "evolve", "run", "--config", cfg_path], cwd=WORK, env=env)
subprocess.run([sys.executable, "-m", "evolve", "status", "--config", cfg_path], cwd=WORK)
subprocess.run([sys.executable, "-m", "evolve", "export", "--config", cfg_path, "--out", state_dir + "/evolved.patch"], cwd=WORK)
shutil.rmtree(state_dir + "/ws", ignore_errors=True)
import pandas as pd
pd.DataFrame([["1_0", "1", True, 1]], columns=["row_id", "game_id", "end_of_game", "score"]).to_parquet(
    "/kaggle/working/submission.parquet", index=False)
'''


def build(variant: str, out: Path) -> Path:
    cfg = json.loads((ROOT / "kaggle" / "config.json").read_text())
    runs = {"submit": SUBMIT_RUN, "dev": DEV_RUN, "evolve": EVOLVE_RUN}
    cells = [
        cell(
            f"# arc26 — Qwen3.8-27B × Raven-style harness ({variant})\n\n"
            "Code: https://github.com/timothy-kira/arc26 (embedded below as a tarball). "
            "Explorer + reasoning DAG + skill sedimentation + offline self-evolution.",
            "markdown",
        ),
        cell("CODE = " + repr(code_blob())),
        cell(SETUP.replace("{cfg!r}", repr(json.dumps(cfg)))),
        cell(VLLM),
        cell(runs[variant]),
    ]
    nb = {
        "metadata": {
            "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {
                "accelerator": cfg["accelerator"],
                "isInternetEnabled": False,
                "isGpuEnabled": True,
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": cells,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "notebook.ipynb").write_text(json.dumps(nb, indent=1))
    slug = cfg["kernel_slug"] + ("" if variant == "submit" else f"-{variant}")
    datasets = [cfg["model_dataset"], cfg["wheelhouse_dataset"], *cfg.get("extra_datasets", [])]
    meta = {
        "id": f"{cfg['username']}/{slug}",
        "title": slug,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": datasets,
        "competition_sources": [cfg["competition"]],
        "kernel_sources": [],
        "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="submit", choices=["submit", "dev", "evolve"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = build(a.variant, Path(a.out or ROOT / "build" / a.variant))
    print(f"built {out}/notebook.ipynb")


if __name__ == "__main__":
    main()
