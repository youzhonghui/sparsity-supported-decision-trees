clean:
	rm output.png
	rm output.pdf

run-demo:
	python dtree.py --csv ./data/fishiris.csv

install:
	pip install -r requirements.txt
