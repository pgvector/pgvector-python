lint:
	pycodestyle . --ignore=E501

publish: clean
	python3 setup.py bdist_wheel --universal
	ls dist
	twine upload dist/*
	make clean

clean:
	rm -rf .pytest_cache build dist pgvector.egg-info
