import argparse
import pydotplus
import pandas as pd
import numpy as np

from dtree import DecisionTree, gini
import math

def cal_metric(predict, gt):
    ngt = np.array(gt)
    npr = np.array(predict)
    
    pos_idx = (ngt == 1)
    neg_idx = (ngt == 0)
    
    return {
        'sensitivity': npr[ngt == 1].mean(),
        'specificity': (1 - npr[ngt == 0]).mean()
    }

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
    tree = DecisionTree()

    str_list = '''
    0:[CRP(mg/L)<5.5] yes=1,no=2,missing=2
	1:[白细胞总数(x10^9/L)<18.6850014] yes=3,no=4,missing=3
		3:[血小板计数(x10^9/L)<171.5] yes=7,no=8,missing=7
			7:[白细胞总数(x10^9/L)<11.8999996] yes=13,no=14,missing=13
				13:[CRP(mg/L)<1.5] yes=19,no=20,missing=20
					19:[中性粒细胞百分比(%)<52.5999985] yes=27,no=28,missing=28
						27:leaf=-0.0121546965
						28:leaf=0.0117647061
					20:[出生时体重(g)<1840] yes=29,no=30,missing=29
						29:leaf=0.0510822535
						30:leaf=0.00118343194
				14:[白细胞总数(x10^9/L)<14.71] yes=21,no=22,missing=21
					21:leaf=-0.0139534893
					22:leaf=0.00118343194
			8:[临床表现异常数<1.5] yes=15,no=16,missing=15
				15:[PCT(ng/ML)<0.375] yes=23,no=24,missing=23
					23:[中性杆状核粒细胞百分比(%)<5] yes=31,no=32,missing=31
						31:leaf=-0.146943495
						32:leaf=0.00930232555
					24:[中性粒细胞百分比(%)<41.0500031] yes=33,no=34,missing=34
						33:leaf=0.0122905029
						34:leaf=-0.00952380989
				16:[出生时体重(g)<1340] yes=25,no=26,missing=25
					25:leaf=-0.00346820801
					26:[出生时体重(g)<1670] yes=35,no=36,missing=35
						35:leaf=0.0171428584
						36:leaf=-0.00116959063
		4:[PCT(ng/ML)<0.13499999] yes=9,no=10,missing=10
			9:leaf=-0.00952380989
			10:[出生时体重(g)<2270] yes=17,no=18,missing=17
				17:leaf=0.084153004
				18:leaf=0.00118343194
	2:[CRP(mg/L)<6.5] yes=5,no=6,missing=6
		5:[白细胞总数(x10^9/L)<12.04] yes=11,no=12,missing=11
			11:leaf=0.0200000014
			12:leaf=-0.00952380989
		6:leaf=0.117241383
    '''
    tree.buildFromString(str_list.split('\n'), data, {0: 1, 1: 2.5})
    print(cal_metric(tree.classify(data), data.values[:, -1]))

    tree.index_to_class = {
        0: '无感染',
        1: '感染'
    }
    tree.savePDF('parse_output.pdf')
    tree.savePNG('parse_output.png')


if __name__ == '__main__':
    main()
