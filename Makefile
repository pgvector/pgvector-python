.PHONY: lint build publish clean

lint:
	pycodestyle . --ignore=E501

build:
	python3 -m build

publish: clean build
	twine upload dist/*

clean:
	rm -rf .pytest_cache dist pgvector.egg-info
