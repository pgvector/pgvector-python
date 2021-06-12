lint:
	pycodestyle . --ignore=E501

publish:
	python3 setup.py bdist_wheel --universal
	ls dist
	# twine upload dist/*
	rm -fr build dist pgvector.egg-info
