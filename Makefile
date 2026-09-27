SOURCES = core catwatch tests

.PHONY: lint typecheck format test verify

lint:
	ruff check $(SOURCES)

typecheck:
	pyright

format:
	python -m black $(SOURCES)
	ruff check --fix $(SOURCES)

test:
	pytest

verify:
	python -c "import core.cv_tools.training, core.cv_tools.onnx_classifier, catwatch.train, catwatch.monitor"
	ruff check $(SOURCES)
	python -m black --check $(SOURCES)
	pytest