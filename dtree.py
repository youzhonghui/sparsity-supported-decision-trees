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
import re

# --------------------------------- utils ----------------------------------

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

def parse(str_list):
    res = {}
    for sentence in str_list:
        washed = sentence.strip()
        if len(washed) == 0:
            continue

        node_index = int(re.match('(\d+):', washed).group(1))
        if 'leaf' in washed:
            val_str = re.search('leaf=([\-\.0-9]+)', washed).group(1)
            res[node_index] = {
                'leaf': True,
                'condition': None,
                'threshold': None,
                'true_node': None,
                'false_node': None,
                'missing_node': None
            }
        else:
            cond, val_str = re.search('\[(.*)\]', washed).group(1).split('<')
            res[node_index] = {
                'leaf': False,
                'condition': cond,
                'threshold': float(val_str),
                'true_node': int(re.search('no=(\d+)', washed).group(1)),     # using >= in our tree node
                'false_node': int(re.search('yes=(\d+)', washed).group(1)),
                'missing_node': int(re.search('missing=(\d+)', washed).group(1))
            }

    return res

# ---------------------- decision tree related -------------------------------

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
    def __init__(self, col=-1, value=None, trueBranch=None, falseBranch=None, results=None, summary=None, missingDirection=True, leaf=False, samples_num=0):
        self.col = col
        self.value = value
        self.missingDirection = missingDirection
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
        self.results = results  # None for nodes, not None for leaves
        self.summary = summary
        self.leaf = leaf        # the sub tree node can be leaf if it is pruned
        self.samples_num = samples_num


class DecisionTree:
    def __init__(self):
        super().__init__()

        self.root = None

    def _growDecisionTreeFrom(self, data, evaluationFunction):
        if len(data) == 0:
            return None

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
            pair = dict(data[self.target_column].value_counts())
            return _DecisionTreeNode(col=bestAttribute[0], value=bestAttribute[1], trueBranch=trueBranch,
                                falseBranch=falseBranch, results=dict((self.index_to_class[k], v) for k, v in pair.items()),
                                summary=dcY, missingDirection=bestAttribute[2], samples_num=len(data))
        else:
            pair = dict(data[self.target_column].value_counts())
            return _DecisionTreeNode(results=dict((self.index_to_class[k], v) for k, v in pair.items()),
                                summary=dcY, leaf=True, samples_num=len(data))

    def addSample(self, sample):
        target = int(sample[self.target_column])
        def _pass(node):
            node.samples_num += 1
            if target not in node.results:
                node.results[target] = 0
            node.results[target] += 1

            if node.leaf:
                return

            if (np.isnan(sample[node.col]) and node.missingDirection) or sample[node.col] >= node.value:
                _pass(node.trueBranch)
            else:
                _pass(node.falseBranch)

        _pass(self.root)

    def buildFromString(self, str_list, data=None, cls_weights={}):
        '''
            make sure str_list from xgboost
        '''
        info = parse(str_list)
        node_dict = {}
        for node_index in range(max(info), -1, -1):
            tmp = info[node_index]
            node = _DecisionTreeNode(
                leaf = tmp['leaf'],
                col = tmp['condition'],
                value = tmp['threshold'],
                results = {0: 0, 1: 0},
                samples_num = 0,
                summary = {'stats' : 0, 'samples' : 0}
            )
            if tmp['true_node'] is not None:
                node.trueBranch = node_dict[tmp['true_node']]
                node.falseBranch = node_dict[tmp['false_node']]
                node.missingDirection = (node_dict[tmp['missing_node']] == node_dict[tmp['true_node']])
            node_dict[node_index] = node

        self.root = node_dict[0]
        if data is not None:
            self._prepare_data(data, cls_weights)
            for idx, row in self.data.iterrows():
                self.addSample(row)

    def _prepare_data(self, data, cls_weights):
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

    def fit(self, data, cls_weights, evaluationFunction=gini):
        """Grows and then returns a binary decision tree.
            evaluationFunction: entropy or gini"""
        self._prepare_data(data, cls_weights)
        self.root = self._growDecisionTreeFrom(self.data, evaluationFunction)
    
    def treeToString(self):
        """Plots the obtained decision tree. """
        def toString(decisionTree, indent=''):
            decisionTree.summary['samples'] = decisionTree.samples_num
            decisionTree.summary['stats'] = '/'.join(['%s: %d' % (self.index_to_class[k], v) for k, v in decisionTree.results.items()])

            if decisionTree.leaf:  # leaf node
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
            decisionTree.summary['samples'] = decisionTree.samples_num
            decisionTree.summary['stats'] = '/'.join(['%s: %d' % (self.index_to_class[k], v) for k, v in decisionTree.results.items()])

            if decisionTree.leaf:  # leaf node
                lsX = [(x, y) for x, y in decisionTree.results.items()]
                lsX.sort()
                szY = ', '.join(['%s: %s' % (x, y) for x, y in lsX])
                dcY = {"name": szY, "parent" : szParent}
                dcSummary = decisionTree.summary
                score = dict([(k, v * self.cls_weights[k]) for k, v in decisionTree.results.items()])
                dcNodes[iSplit].append(['leaf', self.index_to_class[max(score, key=score.get)], szParent, bBranch, missingGoes, dcSummary['stats'],
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

    def _node_explore(self, node, row):
        if node.leaf or ((node.trueBranch is None) and (node.falseBranch is None)):
            score = dict([(k, v * self.cls_weights[k]) for k, v in node.results.items()])
            return max(score, key=score.get)
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
        return self._node_explore(branch, row)

    def classify(self, data):
        """Classifies the observationss according to the tree.
        dataMissing: true or false if data are missing or not. """
        ans = []
        for idx, row in data.iterrows():
            ans.append(self._node_explore(self.root, row))
        return ans

    def _error_rate(self, tree, view_as_leaf=False):
        if tree.leaf or view_as_leaf:
            output = max(tree.results, key=tree.results.get)
            return (1 - (float(tree.results[output]) / tree.samples_num)) * tree.samples_num / tree.samples_num

        tr = self._error_rate(tree.trueBranch) 
        fr = self._error_rate(tree.falseBranch)
        return tr + fr

    def _count_leaf(self, tree):
        if tree.leaf:
            return 1
        return self._count_leaf(tree.trueBranch) + self._count_leaf(tree.falseBranch)

    def _clear_prune(self, tree):
        if tree.trueBranch is None and tree.falseBranch is None:
            tree.leaf = True
        else:
            tree.leaf = False
            self._clear_prune(tree.trueBranch)
            self._clear_prune(tree.falseBranch)

    def prune(self, validation_set, min_alpha=0):
        self._clear_prune(self.root)
        pruned_list = [(None, 0)]
        while not self.root.leaf:
            g = {}
            def _explore(node):
                if node.leaf:
                    return
                g[node] = (self._error_rate(node, view_as_leaf=True) - self._error_rate(node)) / (self._count_leaf(node) - 1)
                if node.trueBranch is not None:
                    _explore(node.trueBranch)
                if node.falseBranch is not None:
                    _explore(node.falseBranch)
            _explore(self.root)

            node_to_prune = min(g, key=g.get)
            pruned_list.append((node_to_prune, g[node_to_prune]))
            node_to_prune.leaf = True
        
        self._clear_prune(self.root)
        error_on_val = []
        ground_truth = validation_set[self.target_column]
        for node_to_prune, _ in pruned_list:
            if node_to_prune is not None:
                node_to_prune.leaf = True

            res = self.classify(validation_set)
            error_rate = np.sum([1 if a != b else 0 for a, b in zip(res, ground_truth)]) / len(ground_truth)
            error_on_val.append(error_rate)

        self._clear_prune(self.root)
        best, idx, alpha = np.inf, 0, 0
        for i, v in enumerate(error_on_val):
            _, alpha = pruned_list[i]
            if alpha < min_alpha:
                idx = i
                continue

            if v <= best:
                idx = i
                best = v
        for i in range(idx + 1):
            node_to_prune, alpha = pruned_list[i]
            if node_to_prune is not None:
                node_to_prune.leaf = True
        
        return alpha
