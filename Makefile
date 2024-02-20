start:
	conda activate BatchPicking
	ulimit -s unlimited

init:
	conda create -n BatchPicking python=3.10
	make start
	pip install -r requirements.txt

pre-commit:
	make format
	make freeze

format:
	isort src
	black src

freeze:
	pipreqs . --encoding=utf8 --force --savepath=requirements.txt --ignore=resources,docs

optimize-joint:
	python src -u optimize -m joint -n examples/toy_instance -t 1800

optimize-sequential:
	python src -u optimize -m sequential -n examples/toy_instance -t 1800

new-exp:
	python src -u experiment -m sequential -ns warehouse_D/data_2023-01-30_00,warehouse_D/data_2023-01-30_04,warehouse_D/data_2023-01-30_08,warehouse_D/data_2023-01-30_12,warehouse_D/data_2023-01-30_16,warehouse_D/data_2023-01-30_20,warehouse_D/data_2023-01-31_00,warehouse_D/data_2023-01-31_04 -t 1800 -l INFO
	python src -u experiment -m sequential -ns warehouse_D/data_2023-01-31_08,warehouse_D/data_2023-01-31_12,warehouse_D/data_2023-01-31_20 -t 1800 -l INFO
	python src -u experiment -m joint -ns warehouse_D/data_2023-01-30_00,warehouse_D/data_2023-01-30_04,warehouse_D/data_2023-01-30_08,warehouse_D/data_2023-01-30_12,warehouse_D/data_2023-01-30_16,warehouse_D/data_2023-01-30_20,warehouse_D/data_2023-01-31_00,warehouse_D/data_2023-01-31_04,warehouse_D/data_2023-01-31_08,warehouse_D/data_2023-01-31_12,warehouse_D/data_2023-01-31_20 -t 1800 -l INFO
	
experiment:
	python src -u experiment -m sequential -ns examples/toy_instance,warehouse_A/data_2023-05-22,warehouse_A/data_2023-05-23,warehouse_A/data_2023-05-24,warehouse_A/data_2023-05-25,warehouse_A/data_2023-05-26,warehouse_A/data_2023-05-27 -t 1800 -l INFO
	python src -u experiment -m sequential -ns warehouse_D/data_2023-01-30_00,warehouse_D/data_2023-01-30_04,warehouse_D/data_2023-01-30_08,warehouse_D/data_2023-01-30_12,warehouse_D/data_2023-01-30_16,warehouse_D/data_2023-01-30_20,warehouse_D/data_2023-01-31_00,warehouse_D/data_2023-01-31_04,warehouse_D/data_2023-01-31_08,warehouse_D/data_2023-01-31_12,warehouse_D/data_2023-01-31_20 -t 1800 -l INFO
	python src -u experiment -m sequential -ns warehouse_B/data_2023-05-23,warehouse_B/data_2023-05-24,warehouse_B/data_2023-05-25,warehouse_B/data_2023-05-26,warehouse_B/data_2023-05-27 -t 1800 -l INFO
	python src -u experiment -m sequential -ns  warehouse_C/2023-09-08_15-00-00_RACK-10,warehouse_C/2023-09-08_15-00-00_RACK-30,warehouse_C/2023-09-09_12-00-00_RACK-4 -t 1800 -l INFO

	python src -u experiment -m joint -ns examples/toy_instance,warehouse_A/data_2023-05-22,warehouse_A/data_2023-05-23,warehouse_A/data_2023-05-24,warehouse_A/data_2023-05-25,warehouse_A/data_2023-05-26,warehouse_A/data_2023-05-27 -t 1800 -l INFO
	python src -u experiment -m joint -ns warehouse_D/data_2023-01-30_00,warehouse_D/data_2023-01-30_04,warehouse_D/data_2023-01-30_08,warehouse_D/data_2023-01-30_12,warehouse_D/data_2023-01-30_16,warehouse_D/data_2023-01-30_20,warehouse_D/data_2023-01-31_00,warehouse_D/data_2023-01-31_04,warehouse_D/data_2023-01-31_08,warehouse_D/data_2023-01-31_12,warehouse_D/data_2023-01-31_20 -t 1800 -l INFO
	python src -u experiment -m joint -ns warehouse_B/data_2023-05-22,warehouse_B/data_2023-05-23,warehouse_B/data_2023-05-24,warehouse_B/data_2023-05-25,warehouse_B/data_2023-05-26,warehouse_B/data_2023-05-27 -t 1800 -l INFO
	python src -u experiment -m joint -ns  warehouse_C/2023-09-08_15-00-00_RACK-10,warehouse_C/2023-09-08_15-00-00_RACK-30,warehouse_C/2023-09-09_12-00-00_RACK-4 -t 1800 -l INFO
	
experiment_all:
	python src -u experiment -m sequential -ns all -t 900 -l INFO

describe:
	python src -u describe -m joint

test:
	make optimize
	make experiment
	make describe

pre-process:
	python src/services/scripts/duplicate_files.py

