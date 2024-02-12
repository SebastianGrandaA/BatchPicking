up:
	conda activate BatchPicking

optimize:
	python src -u optimize -m joint -n examples/toy_instance -t 1500

experiment:
	python src -u experiment -m joint -ns examples/toy_instance -t 1500

pre-process:
	python scripts/duplicate_files.py

freeze:
	pipreqs . --encoding=utf8 --force --savepath=requirements.txt --ignore=resources,docs