PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

MODEL_DEFAULT := hf.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF
MODEL_FROM_ENV := $(shell grep -E '^OLLAMA_MODEL=' .env 2>/dev/null | tail -n1 | cut -d= -f2- | tr -d "\"'")
MODEL ?= $(MODEL_FROM_ENV)

.PHONY: install pull-model setup

install:
	$(PIP) install -r requirements.txt

pull-model:
	@model="$(MODEL)"; \
	if [ -z "$$model" ]; then \
		model="$(MODEL_DEFAULT)"; \
		echo "MODEL not set; defaulting to $$model"; \
	fi; \
	echo "Pulling Ollama model $$model"; \
	ollama pull "$$model"

setup: install pull-model
	@echo "Environment ready. Add/update .env if needed."
