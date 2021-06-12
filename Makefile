lint:
	pycodestyle . --ignore=E501

publish:
	python3 setup.py bdist_wheel --universal
	ls dist
	# twine upload dist/*
	rm -rf build dist pgvector.egg-info
