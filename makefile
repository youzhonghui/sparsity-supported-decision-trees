clean:
	rm output.png
	rm output.pdf

run-demo:
	python main.py --csv ./data/fishiris_with_null.csv

run-string:
	python main.py --csv ./data/tbc.csv

install:
	pip install -r requirements.txt
