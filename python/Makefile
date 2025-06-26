build-wheel:
	uvx --from build pyproject-build

pub-wheel: build-wheel
	uvx twine upload --non-interactive --verbose dist/*
	rm -rf dist
