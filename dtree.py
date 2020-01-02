#  Missing value support was added by Zhonghui You.
#       https://github.com/youzhonghui/sparsity-supported-decision-trees
#
#  The original implementation was written by Michael Dorner.
#       https://github.com/michaeldorner/DecisionTrees
#

import argparse
import math
import csv
from collections import defaultdict
import pydotplus
import numpy as np
import pandas as pd

# --------------------------------- utils ----------------------------------

def uniqueCounts(rows):
    results = {}
    for row in rows:
        #response variable is in the last column
        r = row[-1]
        if r not in results: results[r] = 0
        results[r] += 1
    return results


def variance(rows):
    if len(rows) == 0: return 0
    data = [float(row[len(row) - 1]) for row in rows]
    mean = sum(data) / len(data)

    variance = sum([(d-mean)**2 for d in data]) / len(data)
    return variance


def divideSet(data, column, value, missingDirection):
    splitter = None
    if isinstance(value, int) or isinstance(value, float): # for int and float values
        splitter = lambda candidate : candidate >= value or ((missingDirection == True) and np.isnan(candidate))
    else: # for strings
        splitter = lambda candidate : candidate == value

    index = data[column].apply(splitter)
    list1 = data[index]
    list2 = data[~index]
    return (list1, list2)

# ---------------------- decision tree related -------------------------------

# def entropy(rows):
#     log2 = lambda x: math.log(x)/math.log(2)
#     results = uniqueCounts(rows)

#     entr = 0.0
#     for r in results:
#         p = float(results[r])/len(rows)
#         entr -= p*log2(p)
#     return entr


def gini(targets, weights):
    total = len(targets)
    counts = dict((k, v) for k, v in enumerate(np.bincount(targets.to_numpy())))
    imp = 0.0

    for k in counts:
        w = weights[k] if k in weights else 1
        p = float(counts[k]) / total
        imp += w * p * (1 - p)
    return imp


class _DecisionTreeNode:
    """Binary tree implementation with true and false branch. """
    def __init__(self, col=-1, value=None, trueBranch=None, falseBranch=None, results=None, summary=None, missingDirection=True):
        self.col = col
        self.value = value
        self.missingDirection = missingDirection
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
        self.results = results # None for nodes, not None for leaves
        self.summary = summary


class DecisionTree:
    def __init__(self):
        super().__init__()

        self.root = None

    def _growDecisionTreeFrom(self, data, evaluationFunction):
        if len(data) == 0:
            return _DecisionTreeNode()
        currentScore = evaluationFunction(data[self.target_column], self.cls_weights)

        bestGain = 0.0
        bestAttribute = None
        bestSets = None

        columnCount = len(self.feature_columns)
        for missingDirection in [True, False]:
            for col in self.feature_columns:
                # remove the nan values
                unique = {x for x in set(data[col]) if x == x}
                for value in unique:
                    (set1, set2) = divideSet(data, col, value, missingDirection)

                    p = float(len(set1)) / len(data)
                    gain = currentScore - p * evaluationFunction(set1[self.target_column], self.cls_weights) - \
                                (1-p) * evaluationFunction(set2[self.target_column], self.cls_weights)
                    if gain > bestGain and len(set1) > 0 and len(set2) > 0:
                        bestGain = gain
                        bestAttribute = (col, value, missingDirection)
                        bestSets = (set1, set2)

        stats = []
        for k, v in dict(data[self.target_column].value_counts()).items():
            stats.append('%s: %d' % (self.index_to_class[k], v))
        dcY = {'stats' : '/'.join(stats), 'samples' : '%d' % len(data)}
        if bestGain > 0:
            trueBranch = self._growDecisionTreeFrom(bestSets[0], evaluationFunction)
            falseBranch = self._growDecisionTreeFrom(bestSets[1], evaluationFunction)
            return _DecisionTreeNode(col=bestAttribute[0], value=bestAttribute[1], trueBranch=trueBranch,
                                falseBranch=falseBranch, summary=dcY, missingDirection=bestAttribute[2])
        else:
            pair = dict(data[self.target_column].value_counts())
            return _DecisionTreeNode(results=dict((self.index_to_class[k], v) for k, v in pair.items()), summary=dcY)
    
    def fit(self, data, cls_weights, evaluationFunction=gini):
        """Grows and then returns a binary decision tree.
            evaluationFunction: entropy or gini"""
        self.data = data.copy()
        self.data.columns = self.data.columns.str.strip()

        for col in self.data.columns:
            if self.data[col].dtype == np.dtype('O'):
                self.data[col] = self.data[col].str.strip()

        self.feature_columns = self.data.columns.tolist()[:-1]
        self.target_column = self.data.columns.tolist()[-1]

        elm = list(set(self.data[self.target_column].to_list()))
        self.data[self.target_column] = self.data[self.target_column].apply(lambda x: elm.index(x))
        self.index_to_class = dict((idx, e) for idx, e in enumerate(elm))

        for k in elm:
            if k not in cls_weights:
                cls_weights[k] = 1

        self.cls_weights = dict((elm.index(k), v) for k, v in cls_weights.items())

        self.root = self._growDecisionTreeFrom(self.data, evaluationFunction)
    

    def treeToString(self):
        """Plots the obtained decision tree. """
        def toString(decisionTree, indent=''):
            if decisionTree.results != None:  # leaf node
                lsX = [(x, y) for x, y in decisionTree.results.items()]
                lsX.sort()
                szY = ', '.join(['%s: %s' % (x, y) for x, y in lsX])
                return szY
            else:
                if isinstance(decisionTree.value, int) or isinstance(decisionTree.value, float):
                    decision = '%s >= %s?' % (decisionTree.col, decisionTree.value)
                else:
                    decision = '%s == %s?' % (decisionTree.col, decisionTree.value)

                leftText = 'yes, missing -> ' if decisionTree.missingDirection else 'yes -> '
                rightText = 'no, missing -> ' if not decisionTree.missingDirection else 'no -> '
                trueBranch = indent + leftText + toString(decisionTree.trueBranch, indent + '\t\t')
                falseBranch = indent + rightText + toString(decisionTree.falseBranch, indent + '\t\t')
                return (decision + '\n' + trueBranch + '\n' + falseBranch)

        return toString(self.root)

    def savePDF(self, save_path):
        dot_data = self.dotgraph()
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf(save_path)

    def savePNG(self, save_path):
        dot_data = self.dotgraph()
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_png(save_path)

    def dotgraph(self):
        if self.root is None:
            print('The tree is empty!')
            return 

        dcNodes = defaultdict(list)
        """Plots the obtained decision tree. """
        def toString(iSplit, decisionTree, bBranch, missingGoes, szParent = "null", indent=''):
            if decisionTree.results != None:  # leaf node
                lsX = [(x, y) for x, y in decisionTree.results.items()]
                lsX.sort()
                szY = ', '.join(['%s: %s' % (x, y) for x, y in lsX])
                dcY = {"name": szY, "parent" : szParent}
                dcSummary = decisionTree.summary
                dcNodes[iSplit].append(['leaf', dcY['name'], szParent, bBranch, missingGoes, dcSummary['stats'],
                                        dcSummary['samples']])
                return dcY
            else:
                if isinstance(decisionTree.value, int) or isinstance(decisionTree.value, float):
                        decision = '%s >= %s' % (decisionTree.col, decisionTree.value)
                else:
                        decision = '%s == %s' % (decisionTree.col, decisionTree.value)
                trueBranch = toString(iSplit+1, decisionTree.trueBranch, True, decisionTree.missingDirection == True, decision, indent + '\t\t')
                falseBranch = toString(iSplit+1, decisionTree.falseBranch, False, decisionTree.missingDirection == False, decision, indent + '\t\t')
                dcSummary = decisionTree.summary
                dcNodes[iSplit].append([iSplit+1, decision, szParent, bBranch, missingGoes, dcSummary['stats'],
                                        dcSummary['samples']])
                return

        toString(0, self.root, None, None)
        lsDot = ['digraph Tree {',
                    'node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;',
                    'edge [fontname=helvetica] ;'
        ]
        i_node = 0
        dcParent = {}
        for nSplit in range(len(dcNodes)):
            lsY = dcNodes[nSplit]
            for lsX in lsY:
                iSplit, decision, szParent, bBranch, missingGoes, stats, szSamples =lsX
                if type(iSplit) == int:
                    szSplit = '%d-%s' % (iSplit, decision)
                    dcParent[szSplit] = i_node
                    lsDot.append('%d [label=<%s<br/>%s<br/>samples %s>, fillcolor="#e5813900"] ;' % (i_node,
                                            decision.replace('>=', '&ge;').replace('?', ''),
                                            stats,
                                            szSamples))
                else:
                    lsDot.append('%d [label=<%s<br/>samples %s<br/>class %s>, fillcolor="#e5813900"] ;' % (i_node,
                                            stats,
                                            szSamples,
                                            decision))

                if szParent != 'null':
                    if bBranch:
                        szAngle = '45'
                        szHeadLabel = 'Yes'
                    else:
                        szAngle = '-45'
                        szHeadLabel = 'No'
                    if missingGoes:
                        szHeadLabel += ' or Missing'
                    szSplit = '%d-%s' % (nSplit, szParent)
                    p_node = dcParent[szSplit]
                    lsDot.append('%d -> %d [labeldistance=2.5, labelangle=%s, headlabel="%s"] ;' % (p_node,
                                                        i_node, szAngle, szHeadLabel))

                i_node += 1
        lsDot.append('}')
        dot_data = '\n'.join(lsDot)
        return dot_data

    @classmethod
    def _node_explore(cls, node, row):
        if node.results is not None:    # leaf
            return node.results
        else:
            v = row[node.col]
            branch = None

            if np.isnan(v):
                branch = node.trueBranch if node.missingDirection else node.falseBranch
            elif isinstance(v, int) or isinstance(v, float):
                if v >= node.value:
                    branch = node.trueBranch
                else:
                    branch = node.falseBranch
            else:
                branch = node.trueBranch if v == node.value else node.falseBranch
        return cls._node_explore(branch, row)

    def classify(self, data):
        """Classifies the observationss according to the tree.
        dataMissing: true or false if data are missing or not. """
        ans = []
        for idx, row in data.iterrows():
            ans.append(self._node_explore(self.root, row))
        return ans


# def prune(tree, minGain, evaluationFunction=entropy, notify=False):
#     """Prunes the obtained tree according to the minimal gain (entropy or Gini). """
#     # recursive call for each branch
#     if tree.trueBranch.results == None: prune(tree.trueBranch, minGain, evaluationFunction, notify)
#     if tree.falseBranch.results == None: prune(tree.falseBranch, minGain, evaluationFunction, notify)

#     # merge leaves (potentionally)
#     if tree.trueBranch.results != None and tree.falseBranch.results != None:
#         tb, fb = [], []

#         for v, c in tree.trueBranch.results.items(): tb += [[v]] * c
#         for v, c in tree.falseBranch.results.items(): fb += [[v]] * c

#         p = float(len(tb)) / len(tb + fb)
#         delta = evaluationFunction(tb+fb) - p*evaluationFunction(tb) - (1-p)*evaluationFunction(fb)
#         if delta < minGain:
#             if notify: print('A branch was pruned: gain = %f' % delta)
#             tree.trueBranch, tree.falseBranch = None, None
#             tree.results = uniqueCounts(tb + fb)
