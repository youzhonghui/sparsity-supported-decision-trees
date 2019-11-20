import argparse
import pydotplus

from dtree import loadCSV, DecisionTree, gini, entropy
import math

def main():
    parser = argparse.ArgumentParser(description="csv data file path")
    parser.add_argument(
        "--csv",
        type=str,
        help="The data file path"
    )
    parser.add_argument(
        "--header",
        type=bool,
        default=True,
        help="Whether there is a header in the csv file"
    )
    parser.add_argument(
        "--eval",
        type=str,
        default="gini",
        help="The evaluation function, could be gini or entropy. Default using gini."
    )
    cli_args = parser.parse_args()

    if cli_args.eval not in ['gini', 'entropy']:
        print('The evaluation function should be gini or entropy')
        exit(0)

    dcHeadings, trainingData = loadCSV(cli_args.csv, cli_args.header)
    tree = DecisionTree()
    tree.fit(dcHeadings, trainingData, gini)
    print(tree.treeToString())
    print(tree.classify([5.1, 3.5, math.nan, 1]))
    print(tree.classifyFromDict({
        "SepalLength": 5.1,
        "SepalWidth": 3.5,
        "PetalLength": 11,
        "PetalWidth": 1
    }))

    tree.savePDF('output.pdf')
    tree.savePNG('output.png')


if __name__ == '__main__':
    main()
