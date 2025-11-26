# Orbbec Gemini 335Le: setup and tests

install:
	uv sync

format:
	uv run ruff format .

test-structure:
	uv run pytest tests/orbbec/test_structure.py -q

test-hw:
	uv run pytest tests/orbbec/test_hardware.py -q --run-hardware

capture profile save_dir frames:
	uv run python capture_orbbec.py --profile {{profile}} --save-dir {{save_dir}} --frames {{frames}}
