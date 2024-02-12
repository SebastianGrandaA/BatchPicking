start:
	conda activate BatchPicking

init:
	conda create -n BatchPicking python=3.10
	make start
	pip install -r requirements.txt

format:
	isort src
	black src

optimize:
	python src -u optimize -m joint -n examples/toy_instance -t 1800

experiment:
	python src -u experiment -m joint -ns examples/toy_instance,warehouse_A/data_2023-05-22 -t 1800 -l INFO

experiment_all:
		python src -u experiment -m joint -ns all -t 1800 -l INFO

describe:
	python src -u describe -m joint

test:
	make optimize
	make experiment
	make describe

pre-process:
	python src/services/scripts/duplicate_files.py

freeze:
	pipreqs . --encoding=utf8 --force --savepath=requirements.txt --ignore=resources,docs
