.PHONY: env fetch run scores plots mos test

PY := . .venv/bin/activate && python

env:
	uv venv && uv pip install -r requirements.txt

fetch:
	$(PY) -m code.utils.manifests --validate data/manifests/*.yaml

run:
	$(PY) -m code.pipeline --config configs/benchmark.yml --stage encode_decode

scores:
	$(PY) -m code.pipeline --config configs/benchmark.yml --stage metrics

plots:
	$(PY) scripts/plot_reports.py results/csv/ results/figures/

mos:
	$(PY) scripts/run_subjective_p808_prep.py --manifest data/manifests/afrispeech_dialog_test.yaml

test:
	$(PY) -m code.tests.smoke
