"""Build a self-contained Kaggle notebook that plays ARC-AGI-3 with EverMind Raven + Qwen3.8-27B.

Everything agent-side is Raven: the notebook installs Raven offline (wheelhouse dataset),
serves Qwen3.8-27B (tool calling) and Qwen3-Embedding (for EverOS memory) with vLLM on the
RTX PRO 6000, and runs Raven's own ARC benchmark integration ``benchmarks.arc3.batch``
(embedded below as a tarball from the Raven fork checkout).

Variants:
  submit - competition rerun: play the gateway's hidden games (one scorecard, closed at the
           end). Outside the rerun it only writes the placeholder submission Kaggle requires.
  dev    - play all public games offline; outputs results, Raven sessions and EverOS memory.

Usage: python kaggle/build_notebook.py --variant dev --raven ../evermind-ai/raven
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def arc3_blob(raven: Path) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(raven / "benchmarks" / "__init__.py", arcname="benchmarks/__init__.py")
        tar.add(
            raven / "benchmarks" / "arc3",
            arcname="benchmarks/arc3",
            filter=lambda ti: None if "__pycache__" in ti.name else ti,
        )
    return base64.b64encode(buf.getvalue()).decode()


def cell(src: str, kind: str = "code") -> dict:
    c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


SETUP = r'''
import base64, glob, io, json, os, subprocess, sys, tarfile, time, urllib.request, zipfile
CFG = json.loads(CFG_JSON)
RERUN = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
WORK = "/kaggle/working"
ARC3 = WORK + "/raven_arc3"
os.makedirs(ARC3, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(ARC3_CODE)), mode="r:gz").extractall(ARC3)
COMP = next((p for p in ["/kaggle/input/competitions/" + CFG["competition"], "/kaggle/input/" + CFG["competition"]] if os.path.isdir(p)), None)
print("rerun:", RERUN, "| comp:", COMP, "| inputs:", sorted(glob.glob("/kaggle/input/*/*")))

def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    print(" ".join(map(str, cmd))[:200], "->", r.returncode, (r.stdout + r.stderr)[-1500:])
    return r

def find_dir(slug_part, must):
    for root, dirs, files in os.walk("/kaggle/input"):
        if slug_part in root.lower() and must(root, files):
            return root
    return None
'''

GPU = r'''
r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
print("nvidia-smi:", r.stdout.strip(), r.stderr.strip())
names = [l.lower() for l in r.stdout.splitlines() if l.strip()]
if RERUN or os.path.exists("/kaggle/input"):
    # Hard requirement: this notebook only runs on the competition's RTX PRO 6000.
    assert names and all(CFG["expected_gpu"] in n for n in names), f"expected {CFG['expected_gpu']!r}, got {r.stdout.strip()}"
'''

RAVEN = r'''
# Raven + everos-memory, installed offline into their own venv (keeps vLLM's torch stack separate).
def unpacked(name, marker):
    # A dataset directory may be mounted as-is or as <name>.zip; return a directory holding `marker`.
    for root, dirs, files in os.walk("/kaggle/input"):
        if "raven-offline-wheels" not in root:
            continue
        if os.path.basename(root) == name and any(marker(f) for f in files):
            return root
        if name + ".zip" in files:
            dest = "/tmp/" + name
            zipfile.ZipFile(os.path.join(root, name + ".zip")).extractall(dest)
            return next((r for r, _, fs in os.walk(dest) if any(marker(f) for f in fs)), dest)
    return None

WH = unpacked("wheels", lambda f: f.startswith("raven-") and f.endswith(".whl"))
TIKTOKEN = unpacked("tiktoken_cache", lambda f: True)
print("raven wheelhouse:", WH, "| tiktoken cache:", TIKTOKEN)
RAVEN_PY = "/tmp/ravenv/bin/python"
sh([sys.executable, "-m", "venv", "/tmp/ravenv"])
sh([RAVEN_PY, "-m", "pip", "install", "-q", "--no-index", "--find-links", WH, "--find-links", COMP + "/arc_agi_3_wheels",
    "raven", "everos-memory", "arc-agi"])
sh([RAVEN_PY, "-c", "import raven, raven_everos, everos, arc_agi; print('raven ok')"])
'''

VLLM = r'''
def model_dir(slug):
    return find_dir(slug, lambda root, files: "config.json" in files and any(f.endswith(".safetensors") for f in files))

VLLM_SITE = "/tmp/vllm-site-packages"
def vllm_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = VLLM_SITE
    env.update({"USE_TF": "0", "TRANSFORMERS_NO_TF": "1", "VLLM_NO_USAGE_STATS": "1",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    return env

def install_vllm():
    wh = find_dir("wheelhouse", lambda root, files: "requirements.lock" in files)
    t0 = time.time()
    sh([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", wh, "--requirement", wh + "/requirements.lock",
        "--target", VLLM_SITE, "--upgrade", "--ignore-installed", "--only-binary", ":all:", "--no-compile",
        "--disable-pip-version-check", "--no-warn-conflicts", "-q"])
    print(f"vllm installed in {time.time() - t0:.0f}s")

PROCS = {}
def serve(name, model, port, extra, log):
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model, "--served-model-name", name,
           "--host", "127.0.0.1", "--port", str(port), *extra]
    print("starting:", " ".join(cmd))
    PROCS[name] = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=vllm_env())

def wait_ready(name, port, log, timeout=1500):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if PROCS[name].poll() is not None:
            print(name, "exited:\n", open(log).read()[-4000:]); return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5); break
        except Exception:
            time.sleep(5)
    else:
        print(name, "not ready:\n", open(log).read()[-4000:]); return False
    print(f"{name} ready after {time.time() - t0:.0f}s"); return True

v = CFG["vllm"]
QWEN = model_dir(CFG["qwen_slug"]); EMB = model_dir(CFG["embedding_slug"])
print("qwen:", QWEN, "| embedding:", EMB)
install_vllm()
serve(v["served_model_name"], QWEN, 8000,
      ["--max-model-len", str(v["max_model_len"]), "--gpu-memory-utilization", str(v["gpu_memory_utilization"]),
       "--max-num-seqs", str(v["max_num_seqs"]), *v["extra_args"]], WORK + "/vllm_qwen.log")
LLM_OK = wait_ready(v["served_model_name"], 8000, WORK + "/vllm_qwen.log")
EMB_OK = False
if EMB:
    for runner in (["--runner", "pooling", "--convert", "embed"], ["--task", "embed"]):
        serve("qwen3-embedding", EMB, 8002, [*runner, "--gpu-memory-utilization", str(v["embedding_gpu_util"]),
              "--max-model-len", "8192"], WORK + "/vllm_embed.log")
        EMB_OK = wait_ready("qwen3-embedding", 8002, WORK + "/vllm_embed.log", timeout=600)
        if EMB_OK:
            break
print("LLM_OK", LLM_OK, "EMB_OK", EMB_OK)
'''

RAVEN_CFG = r'''
v = CFG["vllm"]
subject = {
    "providers": {"custom": {"api_base": "http://127.0.0.1:8000/v1", "api_key": "EMPTY", "protocol": "chat"}},
    "agents": {"defaults": {"provider": "custom", "model": v["served_model_name"],
                            "max_tool_iterations": CFG["max_tool_iterations"],
                            "context_window_tokens": v["max_model_len"], "temperature": CFG["temperature"]}},
    "permissions": {"mode": "full"},
}
SUBJECT = WORK + "/subject_runtime.json"
json.dump(subject, open(SUBJECT, "w"), indent=1)
RUN_ENV = {**os.environ,
    "HOME": WORK + "/raven_home", "PYTHONPATH": ARC3,
    "TIKTOKEN_CACHE_DIR": TIKTOKEN or "", "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "NO_PROXY": "127.0.0.1,localhost,gateway", "no_proxy": "127.0.0.1,localhost,gateway",
    "EVEROS_LLM__MODEL": v["served_model_name"], "EVEROS_LLM__BASE_URL": "http://127.0.0.1:8000/v1", "EVEROS_LLM__API_KEY": "EMPTY",
}
if EMB_OK:
    RUN_ENV.update({"EVEROS_EMBEDDING__MODEL": "qwen3-embedding", "EVEROS_EMBEDDING__BASE_URL": "http://127.0.0.1:8002/v1",
                    "EVEROS_EMBEDDING__API_KEY": "EMPTY"})
os.makedirs(RUN_ENV["HOME"], exist_ok=True)
'''

SUBMIT_RUN = r'''
if RERUN:
    sh("curl -s --fail --retry 999 --retry-all-errors --retry-delay 5 --retry-max-time 900 http://gateway:8001/api/games > /dev/null", shell=True)
    cmd = [RAVEN_PY, "-m", "benchmarks.arc3.batch", "--config", SUBJECT, "--out-dir", WORK + "/run", "--games", "all",
           "--conc", str(CFG["concurrency"]), "--gateway", "http://gateway:8001", "--hours", str(CFG["hours"]),
           "--max-actions", str(CFG["max_actions"]), "--workspace", WORK + "/raven_ws"]
    print(" ".join(cmd)); subprocess.run(cmd, cwd=ARC3, env=RUN_ENV)
else:
    import pandas as pd
    pd.DataFrame([["1_0", "1", True, 1]], columns=["row_id", "game_id", "end_of_game", "score"]).to_parquet(WORK + "/submission.parquet", index=False)
'''

DEV_RUN = r'''
cmd = [RAVEN_PY, "-m", "benchmarks.arc3.batch", "--config", SUBJECT, "--out-dir", WORK + "/dev_run",
       "--games", CFG.get("dev_games", "all"), "--conc", str(CFG["concurrency"]), "--hours", str(CFG["dev_hours"]),
       "--max-actions", str(CFG["max_actions"]), "--workspace", WORK + "/raven_ws", "--env-dir", COMP + "/environment_files"]
print(" ".join(cmd)); subprocess.run(cmd, cwd=ARC3, env=RUN_ENV)
print(open(WORK + "/dev_run/summary.json").read() if os.path.exists(WORK + "/dev_run/summary.json") else "no summary")
for p in PROCS.values():
    p.terminate()
import pandas as pd
pd.DataFrame([["1_0", "1", True, 1]], columns=["row_id", "game_id", "end_of_game", "score"]).to_parquet(WORK + "/submission.parquet", index=False)
'''


def build(variant: str, raven: Path, out: Path) -> Path:
    cfg = json.loads((ROOT / "kaggle" / "config.json").read_text())
    runs = {"submit": SUBMIT_RUN, "dev": DEV_RUN}
    need_llm = variant == "dev"
    cells = [
        cell(
            f"# arc26 — EverMind Raven × Qwen3.8-27B on ARC-AGI-3 ({variant})\n\n"
            "Agent: EverMind Raven (fork timothy-kira/Raven, `benchmarks/arc3`). "
            "Model: Qwen3.8-27B FP8 via vLLM on the RTX PRO 6000, memory: Raven's EverOS plugin (local).",
            "markdown",
        ),
        cell("ARC3_CODE = " + repr(arc3_blob(raven)) + "\nCFG_JSON = " + repr(json.dumps(cfg))),
        cell(SETUP),
        cell(GPU),
    ]
    if variant == "submit":
        cells.append(cell("if RERUN:\n" + "\n".join("    " + l for l in (RAVEN + VLLM + RAVEN_CFG).strip().splitlines())))
    else:
        cells += [cell(RAVEN), cell(VLLM), cell(RAVEN_CFG)]
    cells.append(cell(runs[variant]))
    nb = {
        "metadata": {
            "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": cfg["notebook_accelerator"], "isInternetEnabled": False, "isGpuEnabled": True,
                       "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": cells,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "notebook.ipynb").write_text(json.dumps(nb, indent=1))
    slug = cfg["kernel_slug"] + ("" if variant == "submit" else f"-{variant}")
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
        "machine_shape": cfg["machine_shape"],
        "dataset_sources": cfg["dataset_sources"],
        "competition_sources": [cfg["competition"]],
        "kernel_sources": [],
        "model_sources": cfg["model_sources"],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="dev", choices=["submit", "dev"])
    ap.add_argument("--raven", default=str(ROOT.parent / "evermind-ai" / "raven"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = build(a.variant, Path(a.raven), Path(a.out or ROOT / "build" / a.variant))
    print(f"built {out}/notebook.ipynb")


if __name__ == "__main__":
    main()
