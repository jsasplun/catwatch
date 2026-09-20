SOURCES = core catwatch tests

.PHONY: lint typecheck format test verify

lint:
	ruff check $(SOURCES)

typecheck:
	pyright

format:
	ruff check --fix $(SOURCES)

test:
	pytest

verify:
	python -c "import core.training, core.onnx_classifier, catwatch.train, catwatch.monitor"
	ruff check $(SOURCES)
	pytest