"""Build a self-contained Kaggle notebook: MiniMax Code (``mcode exec``) + Qwen3.8-27B play ARC-AGI-3.

The notebook serves Qwen3.8-27B (tool calling, qwen3 reasoning parser) with vLLM on the RTX PRO 6000,
unpacks MiniMax Code + Node 22 from the ``xishengfeng/mcode-offline`` dataset, and runs
``arc_runner/batch.py`` (embedded below together with ``arc_mcp/server.py``): one headless
``mcode exec`` per game, the game exposed to it as an MCP server.

Variants:
  submit - competition rerun: play the gateway's hidden games (one scorecard, closed at the end).
           Outside the rerun it only writes the placeholder submission Kaggle requires.
  dev    - play all public games offline; outputs per-game results, workspaces and learned skills.

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
        for name in ("arc_mcp", "arc_runner", "skills"):
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

r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
print("nvidia-smi:", r.stdout.strip(), r.stderr.strip())
names = [l.lower() for l in r.stdout.splitlines() if l.strip()]
# Hard requirement: only ever run on the competition's RTX PRO 6000.
assert names and all(CFG["expected_gpu"] in n for n in names), f"expected {CFG['expected_gpu']!r}, got {r.stdout.strip()}"
'''

TOOLS = r'''
# The game engine for the MCP server (kernel python); Node 22 and MiniMax Code from their datasets
# (Kaggle's own node is v20, which MiniMax Code does not run on).
sh([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", COMP + "/arc_agi_3_wheels", "arc-agi"])
import zipfile
def stage(part, dest):
    # Copy a mounted dataset to a writable dir, unpacking any archives it holds.
    src = next((r for r, _, _ in os.walk("/kaggle/input") if part in os.path.basename(r)), None)
    assert src, f"dataset {part!r} is not mounted"
    shutil.copytree(src, dest, symlinks=True, dirs_exist_ok=True)
    for a in glob.glob(dest + "/**/*.zip", recursive=True):
        zipfile.ZipFile(a).extractall(os.path.dirname(a))
    for a in glob.glob(dest + "/**/*.tar*", recursive=True):
        tarfile.open(a).extractall(os.path.dirname(a))
    return dest
# Built inside Kaggle from github.com/Timothy-kira/minimax-code by the arc26-mcode-build utility kernel.
stage("arc26-mcode-build", "/tmp/mcode")
NODE = next(p for p in glob.glob("/tmp/mcode/**/bin/node", recursive=True))
MCODE = next(p for p in glob.glob("/tmp/mcode/**/@minimax-ai/code/cli.js", recursive=True))
for f in [NODE] + glob.glob("/tmp/mcode/**/rg", recursive=True) + glob.glob("/tmp/mcode/**/*.node", recursive=True):
    os.chmod(f, 0o755)
ok = sh([NODE, MCODE, "--version"])
assert ok.returncode == 0, "MiniMax Code does not start; stopping before the GPU is used"
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
'''

def batch_cmd(extra: str) -> str:
    return (
        '[sys.executable, CODE + "/arc_runner/batch.py", "--base-url", "http://127.0.0.1:8000/v1", '
        '"--model", CFG["vllm"]["served_model_name"], "--context", str(CFG["vllm"]["max_model_len"]), "--output-limit", str(CFG.get("output_limit", 16384)), '
        '"--conc", str(CFG["concurrency"]), "--max-actions", str(CFG["max_actions"]), '
        '"--max-steps", str(CFG["max_steps"]), "--node", NODE, "--mcode", MCODE, '
        '"--server-python", sys.executable, "--skills-dir", WORK + "/skills", ' + extra + "]"
    )


SUBMIT_RUN = r'''
if RERUN:
    sh("curl -s --fail --retry 999 --retry-all-errors --retry-delay 5 --retry-max-time 900 http://gateway:8001/api/games > /dev/null", shell=True)
    shutil.copytree(CODE + "/skills", WORK + "/skills", dirs_exist_ok=True) if os.path.isdir(CODE + "/skills") else os.makedirs(WORK + "/skills", exist_ok=True)
    cmd = BATCH_SUBMIT
    print(cmd); subprocess.run(cmd, env={**os.environ, "NO_PROXY": "127.0.0.1,localhost,gateway"})
else:
    import pandas as pd
    pd.DataFrame([["1_0", "1", True, 1]], columns=["row_id", "game_id", "end_of_game", "score"]).to_parquet(WORK + "/submission.parquet", index=False)
'''

DEV_RUN = r'''
shutil.copytree(CODE + "/skills", WORK + "/skills", dirs_exist_ok=True) if os.path.isdir(CODE + "/skills") else os.makedirs(WORK + "/skills", exist_ok=True)
cmd = BATCH_DEV
print(cmd); subprocess.run(cmd, env={**os.environ, "NO_PROXY": "127.0.0.1,localhost"})
print(open(WORK + "/dev_run/summary.json").read() if os.path.exists(WORK + "/dev_run/summary.json") else "no summary")
VLLM_PROC.terminate()
for d in glob.glob(WORK + "/dev_run/ws/*/.mcode-data"):
    shutil.rmtree(d, ignore_errors=True)  # keep outputs small: sessions are large, results/notes/skills stay
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
        batch_cmd('"--out-dir", WORK + "/dev_run", "--games", CFG.get("dev_games", "all"), "--env-dir", COMP + "/environment_files", "--hours", str(CFG["dev_hours"])'),
    )
    cells = [
        cell(
            f"# arc26 — MiniMax Code × Qwen3.8-27B on ARC-AGI-3 ({variant})\n\n"
            "Agent: MiniMax Code (`mcode exec`, unmodified) with the game as an MCP server. "
            "Model: Qwen3.8-27B FP8 via vLLM on the RTX PRO 6000. Code: github.com/timothy-kira/arc26.",
            "markdown",
        ),
        cell("CODE_TGZ = " + repr(code_blob()) + "\nCFG_JSON = " + repr(json.dumps(cfg))),
        cell(SETUP),
    ]
    if variant == "submit":
        cells.append(cell("if RERUN:\n" + "\n".join("    " + l for l in (TOOLS + VLLM).strip().splitlines())))
        cells.append(cell(submit))
    else:
        cells += [cell(TOOLS), cell(VLLM), cell(dev)]
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
        "id": f"{cfg['username']}/{slug}", "title": slug, "code_file": "notebook.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "machine_shape": cfg["machine_shape"], "dataset_sources": cfg["dataset_sources"],
        "competition_sources": [cfg["competition"]], "kernel_sources": cfg.get("kernel_sources", []), "model_sources": cfg["model_sources"],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="dev", choices=["submit", "dev"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = build(a.variant, Path(a.out or ROOT / "build" / a.variant))
    print(f"built {out}/notebook.ipynb")


if __name__ == "__main__":
    main()
