"""Kaggle utility kernel (CPU, internet on): build MiniMax Code from github.com/Timothy-kira/minimax-code
with Node 22 and publish it as this kernel's output (mcode-offline.tar.gz: bin/node + pkg/), which the
offline GPU notebook mounts through kernel_sources. Kaggle's own node is v20, which MiniMax Code
cannot run on, and dataset uploads of the bundle are not accepted."""
import os, subprocess, sys, tarfile, urllib.request

W = "/kaggle/working"
NODE_VER = "v22.22.2"
REPO = "https://github.com/Timothy-kira/minimax-code"


def sh(cmd, cwd=None, env=None):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, cwd=cwd, env=env, capture_output=True, text=True)
    print((r.stdout + r.stderr)[-3000:], flush=True)
    if r.returncode:
        sys.exit(f"failed: {cmd}")
    return r.stdout


os.makedirs("/tmp/b", exist_ok=True)
tgz = f"/tmp/b/node-{NODE_VER}-linux-x64.tar.xz"
urllib.request.urlretrieve(f"https://nodejs.org/dist/{NODE_VER}/node-{NODE_VER}-linux-x64.tar.xz", tgz)
sh(f"tar xJf {tgz} -C /tmp/b")
NODE_DIR = f"/tmp/b/node-{NODE_VER}-linux-x64"
env = {**os.environ, "PATH": f"{NODE_DIR}/bin:" + os.environ["PATH"], "COREPACK_ENABLE_DOWNLOAD_PROMPT": "0"}
sh("node --version && corepack enable", env=env)
sh(f"git clone --depth 1 {REPO} /tmp/b/src")
rev = sh("git rev-parse --short HEAD", cwd="/tmp/b/src").strip()
sh("pnpm install --frozen-lockfile", cwd="/tmp/b/src", env=env)
sh("pnpm build", cwd="/tmp/b/src", env={**env, "MCODE_RELEASE_TAG": "v0.5.4"})
sh("node scripts/package-cli-release.mjs v0.5.4 /tmp/b/release", cwd="/tmp/b/src", env=env)
sh("npm install -g --prefix /tmp/b/out/pkg /tmp/b/release/minimax-code-0.5.4.tar.gz --include=optional", env=env)
os.makedirs("/tmp/b/out/bin", exist_ok=True)
sh(f"cp {NODE_DIR}/bin/node /tmp/b/out/bin/node")
sh("/tmp/b/out/bin/node /tmp/b/out/pkg/lib/node_modules/@minimax-ai/code/cli.js --version")
with open("/tmp/b/out/BUILD.txt", "w") as f:
    f.write(f"minimax-code {REPO} @ {rev}\nnode {NODE_VER}\n")
with tarfile.open(f"{W}/mcode-offline.tar.gz", "w:gz") as t:
    t.add("/tmp/b/out", arcname=".")
print("done", os.path.getsize(f"{W}/mcode-offline.tar.gz"))
