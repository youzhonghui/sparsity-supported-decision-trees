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


def divideSet(rows, column, value, missingDirection):
    splittingFunction = None
    if isinstance(value, int) or isinstance(value, float): # for int and float values
        splittingFunction = lambda row : row[column] >= value or ((missingDirection == True) and (row[column] is math.nan))
    else: # for strings
        splittingFunction = lambda row : row[column] == value
    list1 = [row for row in rows if splittingFunction(row)]
    list2 = [row for row in rows if not splittingFunction(row)]
    return (list1, list2)


def loadCSV(file, bHeader):
    """Loads a CSV file and converts all floats and ints into basic datatypes."""
    def convertTypes(s):
        if str.lower(s) == 'null':
            return math.nan
        s = s.strip()
        try:
            return float(s) if '.' in s else int(s)
        except ValueError:
            return s

    reader = csv.reader(open(file, 'r'))
    dcHeader = {}
    if bHeader:
        lsHeader = next(reader)
        for i, szY in enumerate(lsHeader):
                szCol = 'Column %d' % i
                dcHeader[szCol] = str(szY)
    return dcHeader, [[convertTypes(item) for item in row] for row in reader]

# ---------------------- decision tree related -------------------------------

def entropy(rows):
    log2 = lambda x: math.log(x)/math.log(2)
    results = uniqueCounts(rows)

    entr = 0.0
    for r in results:
        p = float(results[r])/len(rows)
        entr -= p*log2(p)
    return entr


def gini(rows):
    total = len(rows)
    counts = uniqueCounts(rows)
    imp = 0.0

    for k1 in counts:
        p1 = float(counts[k1])/total
        for k2 in counts:
            if k1 == k2: continue
            p2 = float(counts[k2])/total
            imp += p1*p2
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
    
    @classmethod
    def _growDecisionTreeFrom(cls, rows, evaluationFunction):
        if len(rows) == 0: return _DecisionTreeNode()
        currentScore = evaluationFunction(rows)

        bestGain = 0.0
        bestAttribute = None
        bestSets = None

        columnCount = len(rows[0]) - 1  # last column is the result/target column
        for missingDirection in [True, False]:
            for col in range(0, columnCount):
                columnValues = [row[col] for row in rows]

                #unique values
                lsUnique = list(set(columnValues))
                if math.nan in lsUnique:
                    lsUnique.remove(math.nan)

                for value in lsUnique:
                    (set1, set2) = divideSet(rows, col, value, missingDirection)

                    # Gain -- Entropy or Gini
                    p = float(len(set1)) / len(rows)
                    gain = currentScore - p*evaluationFunction(set1) - (1-p)*evaluationFunction(set2)
                    if gain>bestGain and len(set1)>0 and len(set2)>0:
                        bestGain = gain
                        bestAttribute = (col, value, missingDirection)
                        bestSets = (set1, set2)

        dcY = {'impurity' : '%.3f' % currentScore, 'samples' : '%d' % len(rows)}
        if bestGain > 0:
            trueBranch = cls._growDecisionTreeFrom(bestSets[0], evaluationFunction)
            falseBranch = cls._growDecisionTreeFrom(bestSets[1], evaluationFunction)
            return _DecisionTreeNode(col=bestAttribute[0], value=bestAttribute[1], trueBranch=trueBranch,
                                falseBranch=falseBranch, summary=dcY, missingDirection=bestAttribute[2])
        else:
            return _DecisionTreeNode(results=uniqueCounts(rows), summary=dcY)
    
    def fit(self, rows, evaluationFunction=entropy):
        """Grows and then returns a binary decision tree.
            evaluationFunction: entropy or gini"""
        self.root = self._growDecisionTreeFrom(rows, evaluationFunction)
    

    def treeToString(self, dcHeadings):
        """Plots the obtained decision tree. """
        def toString(decisionTree, indent=''):
            if decisionTree.results != None:  # leaf node
                lsX = [(x, y) for x, y in decisionTree.results.items()]
                lsX.sort()
                szY = ', '.join(['%s: %s' % (x, y) for x, y in lsX])
                return szY
            else:
                szCol = 'Column %s' % decisionTree.col
                if szCol in dcHeadings:
                    szCol = dcHeadings[szCol]
                if isinstance(decisionTree.value, int) or isinstance(decisionTree.value, float):
                    decision = '%s >= %s?' % (szCol, decisionTree.value)
                else:
                    decision = '%s == %s?' % (szCol, decisionTree.value)

                leftText = 'yes, missing -> ' if decisionTree.missingDirection else 'yes -> '
                rightText = 'no, missing -> ' if not decisionTree.missingDirection else 'no -> '
                trueBranch = indent + leftText + toString(decisionTree.trueBranch, indent + '\t\t')
                falseBranch = indent + rightText + toString(decisionTree.falseBranch, indent + '\t\t')
                return (decision + '\n' + trueBranch + '\n' + falseBranch)

        return toString(self.root)

    def savePDF(self, save_path, dcHeadings):
        dot_data = self.dotgraph(dcHeadings)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf(save_path)

    def savePNG(self, save_path, dcHeadings):
        dot_data = self.dotgraph(dcHeadings)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_png(save_path)

    def dotgraph(self, dcHeadings):
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
                dcNodes[iSplit].append(['leaf', dcY['name'], szParent, bBranch, missingGoes, dcSummary['impurity'],
                                        dcSummary['samples']])
                return dcY
            else:
                szCol = 'Column %s' % decisionTree.col
                if szCol in dcHeadings:
                        szCol = dcHeadings[szCol]
                if isinstance(decisionTree.value, int) or isinstance(decisionTree.value, float):
                        decision = '%s >= %s' % (szCol, decisionTree.value)
                else:
                        decision = '%s == %s' % (szCol, decisionTree.value)
                trueBranch = toString(iSplit+1, decisionTree.trueBranch, True, decisionTree.missingDirection == True, decision, indent + '\t\t')
                falseBranch = toString(iSplit+1, decisionTree.falseBranch, False, decisionTree.missingDirection == False, decision, indent + '\t\t')
                dcSummary = decisionTree.summary
                dcNodes[iSplit].append([iSplit+1, decision, szParent, bBranch, missingGoes, dcSummary['impurity'],
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
                iSplit, decision, szParent, bBranch, missingGoes, szImpurity, szSamples =lsX
                if type(iSplit) == int:
                    szSplit = '%d-%s' % (iSplit, decision)
                    dcParent[szSplit] = i_node
                    lsDot.append('%d [label=<%s<br/>impurity %s<br/>samples %s>, fillcolor="#e5813900"] ;' % (i_node,
                                            decision.replace('>=', '&ge;').replace('?', ''),
                                            szImpurity,
                                            szSamples))
                else:
                    lsDot.append('%d [label=<impurity %s<br/>samples %s<br/>class %s>, fillcolor="#e5813900"] ;' % (i_node,
                                            szImpurity,
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

            if v is math.nan:
                branch = node.trueBranch if node.missingDirection else node.falseBranch
            elif isinstance(v, int) or isinstance(v, float):
                if v >= node.value:
                    branch = node.trueBranch
                else:
                    branch = node.falseBranch
            else:
                branch = node.trueBranch if v == node.value else node.falseBranch
        return cls._node_explore(branch, row)

    def classify(self, row):
        """Classifies the observationss according to the tree.
        dataMissing: true or false if data are missing or not. """

        return self._node_explore(self.root, row)


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


# def classify(observations, tree, dataMissing=False):
#     """Classifies the observationss according to the tree.
#     dataMissing: true or false if data are missing or not. """

#     def classifyWithoutMissingData(observations, tree):
#         if tree.results != None:  # leaf
#             return tree.results
#         else:
#             v = observations[tree.col]
#             branch = None
#             if isinstance(v, int) or isinstance(v, float):
#                 if v >= tree.value: branch = tree.trueBranch
#                 else: branch = tree.falseBranch
#             else:
#                 if v == tree.value: branch = tree.trueBranch
#                 else: branch = tree.falseBranch
#         return classifyWithoutMissingData(observations, branch)


#     def classifyWithMissingData(observations, tree):
#         if tree.results != None:  # leaf
#             return tree.results
#         else:
#             v = observations[tree.col]
#             if v == None:
#                 tr = classifyWithMissingData(observations, tree.trueBranch)
#                 fr = classifyWithMissingData(observations, tree.falseBranch)
#                 tcount = sum(tr.values())
#                 fcount = sum(fr.values())
#                 tw = float(tcount)/(tcount + fcount)
#                 fw = float(fcount)/(tcount + fcount)
#                 result = defaultdict(int) # Problem description: http://blog.ludovf.net/python-collections-defaultdict/
#                 for k, v in tr.items(): result[k] += v*tw
#                 for k, v in fr.items(): result[k] += v*fw
#                 return dict(result)
#             else:
#                 branch = None
#                 if isinstance(v, int) or isinstance(v, float):
#                     if v >= tree.value: branch = tree.trueBranch
#                     else: branch = tree.falseBranch
#                 else:
#                     if v == tree.value: branch = tree.trueBranch
#                     else: branch = tree.falseBranch
#             return classifyWithMissingData(observations, branch)

#     # function body
#     if dataMissing:
#         return classifyWithMissingData(observations, tree)
#     else:
#         return classifyWithoutMissingData(observations, tree)


# def plot(decisionTree, dcHeadings):
#     """Plots the obtained decision tree. """
#     def toString(decisionTree, indent=''):
#         if decisionTree.results != None:  # leaf node
#             lsX = [(x, y) for x, y in decisionTree.results.items()]
#             lsX.sort()
#             szY = ', '.join(['%s: %s' % (x, y) for x, y in lsX])
#             return szY
#         else:
#             szCol = 'Column %s' % decisionTree.col
#             if szCol in dcHeadings:
#                 szCol = dcHeadings[szCol]
#             if isinstance(decisionTree.value, int) or isinstance(decisionTree.value, float):
#                 decision = '%s >= %s?' % (szCol, decisionTree.value)
#             else:
#                 decision = '%s == %s?' % (szCol, decisionTree.value)

#             leftText = 'yes, missing -> ' if decisionTree.missingDirection else 'yes -> '
#             rightText = 'no, missing -> ' if not decisionTree.missingDirection else 'no -> '
#             trueBranch = indent + leftText + toString(decisionTree.trueBranch, indent + '\t\t')
#             falseBranch = indent + rightText + toString(decisionTree.falseBranch, indent + '\t\t')
#             return (decision + '\n' + trueBranch + '\n' + falseBranch)

#     print(toString(decisionTree))

