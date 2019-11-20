clean:
	rm output.png
	rm output.pdf

run-demo:
	python main.py --csv ./data/fishiris_with_null.csv

install:
	pip install -r requirements.txt
