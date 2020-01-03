clean:
	rm output.png
	rm output.pdf

run-demo:
	python main.py --csv ./data/fishiris_with_null.csv

run-string:
	python main.py --csv ./data/tbc.csv

run-parse:
	python parse-demo.py --csv ./data/all_wo_last_week.csv

install:
	pip install -r requirements.txt
