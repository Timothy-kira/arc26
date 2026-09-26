# arc26 workflow. Kaggle auth: put your token in .kaggle/access_token (gitignored).
PY      := .venv/bin/python
KAGGLE  := KAGGLE_API_TOKEN=$$(cat .kaggle/access_token) .venv/bin/kaggle
CFG     := kaggle/config.json
USER    := $(shell python3 -c "import json;print(json.load(open('$(CFG)'))['username'])")
SLUG    := $(shell python3 -c "import json;print(json.load(open('$(CFG)'))['kernel_slug'])")

.PHONY: setup data test baseline mock notebooks push-dev push-evolve push-submit status-% pull-%

setup:            ## venv + offline engine wheels + dev deps
	uv venv --python 3.12 .venv
	uv pip install -p .venv kaggle pandas pyarrow httpx pillow pytest pytest-asyncio ruff
	$(MAKE) data
	uv pip install -p .venv --no-index --find-links data/arc_agi_3_wheels arc_agi arcengine
	uv pip install -p .venv -e . --no-deps

data:             ## download competition data (wheels, 25 public games, templates)
	mkdir -p data && cd data && KAGGLE_API_TOKEN=$$(cat ../.kaggle/access_token) ../.venv/bin/kaggle competitions download -c arc-prize-2026-arc-agi-3 -p . && unzip -q -o *.zip && rm -f *.zip

test:
	$(PY) -m pytest -q

baseline:         ## pure explorer on all public games (CPU, ~30s)
	$(PY) -m arc_harness.run --mode offline --llm none --max-actions 600 --out runs/baseline

mock:             ## full pipeline with a scripted LLM (CPU)
	$(PY) -m arc_harness.run --mode offline --llm mock --max-actions 600 --out runs/mock

notebooks:
	for v in submit dev evolve; do $(PY) kaggle/build_notebook.py --variant $$v; done

push-dev: notebooks       ## Qwen on all public games (RTX 6000, uses GPU quota)
	$(KAGGLE) kernels push -p build/dev
push-evolve: notebooks    ## one evolver session (RTX 6000, uses GPU quota)
	$(KAGGLE) kernels push -p build/evolve
push-submit: notebooks    ## submission kernel; then click "Submit to Competition" on Kaggle
	$(KAGGLE) kernels push -p build/submit

status-%:
	$(KAGGLE) kernels status $(USER)/$(SLUG)$(if $(filter submit,$*),,-$*)
pull-%:
	mkdir -p runs/kaggle_$* && $(KAGGLE) kernels output $(USER)/$(SLUG)$(if $(filter submit,$*),,-$*) -p runs/kaggle_$*
