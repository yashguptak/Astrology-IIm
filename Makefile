.PHONY: setup setup-gpu pipeline train train-colab merge serve streamlit test lint clean

# --- Local development ---

setup:
	pip install -r requirements-local.txt

setup-gpu:
	pip install -r requirements-gpu.txt
	pip install -r requirements-local.txt

pipeline:
	python run_pipeline.py

# --- Training ---

train:
	python -m src.training.train --config configs/training.yaml

train-resume:
	python -m src.training.train --config configs/training.yaml --resume_from_checkpoint $(checkpoint)

merge:
	python -m src.training.merge

# --- Inference ---

serve:
	python -m src.inference.interactive_chat

streamlit:
	streamlit run streamlit_app.py

# --- Cloud ---

train-colab:
	@echo "Open scripts/train_colab.ipynb in Google Colab"
	@echo "Upload project to Google Drive first"

train-docker:
	bash scripts/train_docker.sh

# --- Quality ---

lint:
	ruff check src/

test:
	pytest tests/ -v

# --- Clean ---

clean:
	rm -rf training/checkpoints/*
	rm -rf training/logs/*
	rm -rf training/merged/*
	rm -rf data/processed/*
	rm -rf __pycache__
	rm -rf src/**/__pycache__
	rm -rf .pytest_cache
