import argparse
import pydotplus
import pandas as pd
import numpy as np

from dtree import DecisionTree, gini
import math

def main():
    parser = argparse.ArgumentParser(description="csv data file path")
    parser.add_argument(
        "--csv",
        type=str,
        help="The data file path"
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

    data = pd.read_csv(cli_args.csv)
    train = data.sample(frac=0.75, random_state=0)
    test = pd.concat([train, data]).drop_duplicates(keep=False)

    class_weights = {
        'setosa': 1,
        'versicolor': 1,
        'virginica': 1
    }
    tree = DecisionTree()
    tree.fit(train, class_weights, gini)
    # print(tree._error_rate(tree.root))
    print(tree._count_leaf(tree.root))
    # tree.prune(test, 0.0)

    print(tree.treeToString())

    data = pd.DataFrame([[5.1, 3.5, np.nan, 1]], columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'])
    print(tree.classify(data))

    tree.savePDF('output.pdf')
    tree.savePNG('output.png')


if __name__ == '__main__':
    main()
