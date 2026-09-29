"""Build a self-contained Kaggle notebook that plays ARC-AGI-3 with the hand-written agent loop.

Variants:
  submit - competition rerun: vLLM serves Qwen3.8-27B on the RTX PRO 6000 and ``arc_agent.run`` plays
           the gateway's hidden games (one scorecard, closed at the end). Outside the rerun it only
           writes the placeholder submission Kaggle requires.
  dev    - the same on the public games offline, with per-game logs and graphs in the output.
  api    - CPU notebook with internet: the hosted model (Dots) instead of vLLM, for long evaluation
           runs that a restarting container cannot hold.

Usage: python kaggle/build_notebook.py --variant dev
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def code_blob() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in ("arc_agent", "arc_eval"):
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
import base64, glob, io, json, os, shutil, subprocess, sys, tarfile, time, urllib.request
CFG = json.loads(CFG_JSON)
RERUN = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
WORK = "/kaggle/working"
CODE = WORK + "/arc26"
os.makedirs(CODE, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE_TGZ)), mode="r:gz").extractall(CODE)
COMP = next(p for p in ["/kaggle/input/competitions/" + CFG["competition"], "/kaggle/input/" + CFG["competition"]] if os.path.isdir(p))
print("rerun:", RERUN, "| comp:", COMP, "| inputs:", sorted(glob.glob("/kaggle/input/*/*")))

def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    print(str(cmd)[:200], "->", r.returncode, (r.stdout + r.stderr)[-1500:])
    return r

def find_dir(part, must):
    for root, dirs, files in os.walk("/kaggle/input"):
        if part in root.lower() and must(root, files):
            return root
    return None

if NEED_GPU:
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
    print("nvidia-smi:", r.stdout.strip(), r.stderr.strip())
    names = [l.lower() for l in r.stdout.splitlines() if l.strip()]
    # Hard requirement: only ever run on the competition's RTX PRO 6000.
    assert names and all(CFG["expected_gpu"] in n for n in names), f"expected {CFG['expected_gpu']!r}, got {r.stdout.strip()}"
'''

TOOLS = r'''
# The game engine, from the competition's wheels.
sh([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", COMP + "/arc_agi_3_wheels", "arc-agi"])
'''

VLLM = r'''
VLLM_SITE = "/tmp/vllm-site-packages"
def cuda_link_dir():
    """A dir with libcuda.so for FlashInfer's JIT link step (the image has only the driver's libcuda.so.1)."""
    d = "/tmp/cuda-link"
    os.makedirs(d, exist_ok=True)
    cands = glob.glob("/usr/local/cuda*/lib64/stubs/libcuda.so") + glob.glob("/usr/local/cuda*/targets/*/lib/stubs/libcuda.so") \
        + glob.glob("/usr/lib/x86_64-linux-gnu/libcuda.so*") + glob.glob("/usr/local/nvidia/lib64/libcuda.so*")
    if cands and not os.path.exists(d + "/libcuda.so"):
        os.symlink(cands[0], d + "/libcuda.so")
    print("libcuda for linking:", cands[:1])
    return d

def vllm_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = VLLM_SITE
    env["LIBRARY_PATH"] = ":".join(filter(None, [cuda_link_dir(), env.get("LIBRARY_PATH")]))
    env.update({"USE_TF": "0", "TRANSFORMERS_NO_TF": "1", "VLLM_NO_USAGE_STATS": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    return env

wh = find_dir("wheelhouse", lambda root, files: "requirements.lock" in files)
t0 = time.time()
sh([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", wh, "--requirement", wh + "/requirements.lock",
    "--target", VLLM_SITE, "--upgrade", "--ignore-installed", "--only-binary", ":all:", "--no-compile",
    "--disable-pip-version-check", "--no-warn-conflicts", "-q"])
print(f"vllm installed in {time.time() - t0:.0f}s")
# vLLM's streaming qwen3_coder tool parser crashes (and drops the stream) when a parameter value parses
# to a Python set: serialise sets as lists instead.
for tp in glob.glob(VLLM_SITE + "/vllm/tool_parsers/qwen3coder_tool_parser.py") + glob.glob(VLLM_SITE + "/vllm/**/qwen3coder_tool_parser.py", recursive=True):
    src = open(tp).read()
    if "default=list" not in src:
        open(tp, "w").write(src.replace("json.dumps(converted_value, ensure_ascii=False)", "json.dumps(converted_value, ensure_ascii=False, default=list)"))
        print("patched", tp)
v = CFG["vllm"]
QWEN = find_dir(CFG["qwen_slug"], lambda root, files: "config.json" in files and any(f.endswith(".safetensors") for f in files))
cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", QWEN, "--served-model-name", v["served_model_name"],
       "--host", "127.0.0.1", "--port", "8000", "--max-model-len", str(v["max_model_len"]),
       "--gpu-memory-utilization", str(v["gpu_memory_utilization"]), "--max-num-seqs", str(v["max_num_seqs"]), *v["extra_args"]]
print("starting:", " ".join(cmd))
VLLM_PROC = subprocess.Popen(cmd, stdout=open(WORK + "/vllm.log", "w"), stderr=subprocess.STDOUT, env=vllm_env())
t0 = time.time()
while time.time() - t0 < 1500:
    assert VLLM_PROC.poll() is None, "vLLM exited:\n" + "".join(
        l[-300:] for l in open(WORK + "/vllm.log", errors="replace") if " ERROR " in l or "error:" in l.lower())[-6000:]
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5); break
    except Exception:
        time.sleep(5)
print(f"vLLM ready after {time.time() - t0:.0f}s")
print("".join(l for l in open(WORK + "/vllm.log", errors="replace") if "KV cache" in l or "Maximum concurrency" in l))

def echo_vllm_errors(path=WORK + "/vllm.log", limit=300):
    """Copy vLLM's ERROR / exception lines into the notebook log, which streams while the run is live."""
    import threading
    def run():
        n = 0
        with open(path, errors="replace") as f:
            f.seek(0, 2)
            while n < limit:
                line = f.readline()
                if not line:
                    time.sleep(2); continue
                if " ERROR " in line or "Traceback" in line or "Exception" in line:
                    n += 1; print("[vllm]", line.rstrip()[-400:], flush=True)
    threading.Thread(target=run, daemon=True).start()

echo_vllm_errors()
'''

API_RUN = r'''
# Long evaluation runs on Kaggle (CPU + internet) with the hosted model.
A = CFG["api"]
key = None
try:
    from kaggle_secrets import UserSecretsClient
    key = UserSecretsClient().get_secret("DOTS_API_KEY")
except Exception as exc:
    print("no Kaggle secret DOTS_API_KEY:", type(exc).__name__)
if not key:  # fallback: the private dataset arc26-secrets
    kf = next((os.path.join(r, "dots_api_key") for r, _, fs in os.walk("/kaggle/input") if "dots_api_key" in fs), None)
    key = open(kf).read().strip() if kf else None
assert key, "no API key: add the Kaggle secret DOTS_API_KEY or mount the private dataset arc26-secrets"
os.makedirs("/tmp/secrets", exist_ok=True)
open("/tmp/secrets/key", "w").write(key); os.chmod("/tmp/secrets/key", 0o600)
cmd = [sys.executable, "-m", "arc_agent.run", "--out-dir", WORK + "/run", "--games", A["games"], "--k", str(A.get("k", 1)),
       "--conc", str(A["conc"]), "--env-dir", COMP + "/environment_files", "--base-url", A["base_url"], "--model", A["model"],
       "--api-key-file", "/tmp/secrets/key", "--game-seconds", str(A["game_seconds"]), "--hours", str(A["hours"]),
       "--call-seconds", str(A.get("call_seconds", 180))] + ([] if A.get("thinking", True) else ["--no-thinking"]) \
      + ["--think-policy", A.get("think_policy", "model")]
print(cmd); subprocess.run(cmd, cwd=CODE, env={**os.environ, "PYTHONPATH": CODE})
import pandas as pd
pd.DataFrame([["1_0", "1", True, 1]], columns=["row_id", "game_id", "end_of_game", "score"]).to_parquet(WORK + "/submission.parquet", index=False)
'''


def batch_cmd(extra: str) -> str:
    return (
        '[sys.executable, "-m", "arc_agent.run", "--base-url", "http://127.0.0.1:8000/v1", '
        '"--model", CFG["vllm"]["served_model_name"], "--conc", str(CFG["concurrency"]), '
        '"--max-actions", str(CFG["max_actions"]), "--game-seconds", str(CFG["game_seconds"]), '
        '"--call-seconds", str(CFG["call_seconds"]), ' + extra + "]"
        ' + ([] if CFG.get("thinking", True) else ["--no-thinking"]) + ["--think-policy", CFG.get("think_policy", "model")]'
    )


SUBMIT_RUN = r'''
if RERUN:
    sh("curl -s --fail --retry 999 --retry-all-errors --retry-delay 5 --retry-max-time 900 http://gateway:8001/api/games > /dev/null", shell=True)
    cmd = BATCH_SUBMIT
    print(cmd); subprocess.run(cmd, cwd=CODE, env={**os.environ, "PYTHONPATH": CODE, "NO_PROXY": "127.0.0.1,localhost,gateway"})
else:
    import pandas as pd
    pd.DataFrame([["1_0", "1", True, 1]], columns=["row_id", "game_id", "end_of_game", "score"]).to_parquet(WORK + "/submission.parquet", index=False)
'''

DEV_RUN = r'''
cmd = BATCH_DEV
print(cmd); subprocess.run(cmd, cwd=CODE, env={**os.environ, "PYTHONPATH": CODE, "NO_PROXY": "127.0.0.1,localhost"})
VLLM_PROC.terminate()
import pandas as pd
pd.DataFrame([["1_0", "1", True, 1]], columns=["row_id", "game_id", "end_of_game", "score"]).to_parquet(WORK + "/submission.parquet", index=False)
'''


def build(variant: str, out: Path) -> Path:
    cfg = json.loads((ROOT / "kaggle" / "config.json").read_text())
    submit = SUBMIT_RUN.replace(
        "BATCH_SUBMIT",
        batch_cmd('"--out-dir", WORK + "/run", "--games", "all", "--gateway", "http://gateway:8001", "--hours", str(CFG["hours"])'),
    )
    dev = DEV_RUN.replace(
        "BATCH_DEV",
        batch_cmd('"--out-dir", WORK + "/run", "--games", CFG.get("dev_games", "all"), "--env-dir", COMP + "/environment_files", "--hours", str(CFG["dev_hours"])'),
    )
    cells = [
        cell(
            f"# arc26 — hand-written agent loop on ARC-AGI-3 ({variant})\n\n"
            "One LLM call per action from a fresh context: reasoning graph + handoff + checked prediction + "
            "frame diff. Model: Qwen3.8-27B FP8 via vLLM on the RTX PRO 6000 (api variant: hosted model). "
            "Code: github.com/timothy-kira/arc26.",
            "markdown",
        ),
        cell("CODE_TGZ = " + repr(code_blob()) + "\nCFG_JSON = " + repr(json.dumps(cfg))
             + f"\nNEED_GPU = {variant != 'api'}"),
        cell(SETUP),
    ]
    if variant == "submit":
        cells.append(cell("if RERUN:\n" + "\n".join("    " + l for l in (TOOLS + VLLM).strip().splitlines())))
        cells.append(cell(submit))
    elif variant == "api":
        cells += [cell(TOOLS), cell(API_RUN)]
    else:
        cells += [cell(TOOLS), cell(VLLM), cell(dev)]
    nb = {
        "metadata": {
            "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "none" if variant == "api" else cfg["notebook_accelerator"],
                       "isInternetEnabled": variant == "api", "isGpuEnabled": variant != "api",
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
        "id": f"{cfg['username']}/{slug}", "title": slug, "code_file": "notebook.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": variant != "api", "enable_tpu": False,
        "enable_internet": variant == "api",
        "dataset_sources": cfg["api"].get("dataset_sources", []) if variant == "api" else cfg["dataset_sources"],
        "competition_sources": [cfg["competition"]], "kernel_sources": [],
        "model_sources": [] if variant == "api" else cfg["model_sources"],
    }
    if variant != "api":
        meta["machine_shape"] = cfg["machine_shape"]
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="dev", choices=["submit", "dev", "api"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = build(a.variant, Path(a.out or ROOT / "build" / a.variant))
    print(f"built {out}/notebook.ipynb")


if __name__ == "__main__":
    main()
