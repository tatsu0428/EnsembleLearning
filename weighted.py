import numpy as np
import support
from zeror import ZeroRule
from dstump import DecisionStump
from pruning import PrunedTree, getscore, criticalscore

# 重み付きのMetrics関数
def w_gini(y, weight):
    i = y.argmax(axis=1)
    clz = set(i)
    score = 0.0
    for val in clz:
        p = weight[i==val].sum()
        score += p ** 2
    return 1.0 - score

def w_infogain(y, weight):
    i = y.argmax(axis=1)
    clz = set(i)
    score = 0.0
    for val in clz:
        p = weight[i==val].sum()
        if p != 0:
            score += p * np.log2(p)
    return -score

# 重み付きの葉となるモデル
class WeightedZeroRule(ZeroRule):
    def fit(self, X, y, weight):
        # 重み付き平均を取る
        self.r = np.average(y, axis=0, weights=weight)
        return self

class WeightedDecisionStump(DecisionStump):
    def __init__(self, metric=w_infogain, leaf=WeightedZeroRule):
        super().__init__(metric=metric, leaf=leaf)
        self.weight = None

    def make_loss(self, y1, y2, l, r):
        # yをy1とy2で分割したときのMetrics関数の重み付き合計を返す
        if y1.shape[0] == 0 or y2.shape[0] == 0:
            return np.inf
        # Metrics関数に渡す左右のデータの重み
        w1 = self.weight[l] / np.sum(self.weight[l]) # 重みの正規化
        w2 = self.weight[r] / np.sum(self.weight[r]) # 重みの正規化
        total = y1.shape[0] + y2.shape[0]
        m1 = self.metric(y1, w1) * (y1.shape[0]/total)
        m2 = self.metric(y2, w2) * (y2.shape[0]/total)
        return m1 + m2

    def fit(self, X, y, weight):
        # 左右の葉を作成する
        self.weight = weight # 重みを保存
        self.left = self.leaf()
        self.right = self.leaf()
        # データを左右の葉に振り分ける
        left, right = self.split_tree(X, y)
        # 重みを付けて左右の葉を学習させる
        if len(left) > 0:
            self.left.fit(X[left], y[left], weight[left]/np.sum(weight[left])) # 重みの正規化
        if len(right) > 0:
            self.right.fit(X[right], y[right], weight[right]/np.sum(weight[right])) # 重みの正規化
        return self

class WeightedDecisionTree(PrunedTree, WeightedDecisionStump):
    def __init__(self, max_depth=5, metric=w_gini, leaf=WeightedZeroRule, depth=1):
        super().__init__(max_depth=max_depth, metric=metric, leaf=leaf, depth=depth)
        self.weight = None

    def get_node(self):
        # 新しくノードを作成する
        return WeightedDecisionTree(max_depth=self.max_depth, metric=self.metric, leaf=self.leaf, depth=self.depth + 1)

    def fit(self, X, y, weight):
        self.weight = weight # 重みを保存
        # 深さ=1，根のノードのときのみ
        if self.depth == 1 and self.prunfnc is not None:
            # プルーニングに使用するデータ
            X_t, y_t = X, y

        # 決定木の学習・・・"critical"プルーニング時は木の分割のみ
        self.left = self.leaf()
        self.right = self.leaf()
        left, right = self.split_tree(X, y)
        if self.depth < self.max_depth:
            if len(left) > 0:
                self.left = self.get_node()
            if len(right) > 0:
                self.right = self.get_node()
        if self.depth < self.max_depth or self.prunfnc != "critical":
            # 重みを付けて左右の枝を学習させる
            if len(left) > 0:
                self.left.fit(X[left], y[left], weight[left]/np.sum(weight[left])) # 重みの正規化
            if len(right) > 0:
                self.right.fit(X[right], y[right], weight[right]/np.sum(weight[right])) # 重みの正規化

        # 深さ=1，根のノードの時のみ
        if self.depth == 1 and self.prunfnc is not None:
            if self.prunfnc == "critical":
                # 学習時のMetrics関数のスコアを取得する
                score = []
                getscore(self, score)
                # スコアから残す枝の最大スコアを計算
                i = int(round(len(score) * self.critical))
                score_max = sorted(score)[i]
                # プルーニングを行う
                criticalscore(self, score_max)
                # 葉を学習させる
                self.fit_leaf(X, y, weight)

        return self

    def fit_leaf(self, X, y, weight):
        # 説明変数から分割した左右のインデックスを取得
        feat = X[:, self.feat_index]
        val = self.feat_val
        l, r = self.make_split(feat, val)
        # 葉のみを学習させる
        if len(l) > 0:
            if isinstance(self.left, PrunedTree):
                self.left.fit_leaf(X[l], y[l], weight[l])
            else:
                self.left.fit(X[l], y[l], weight[l])
        if len(r) > 0:
            if isinstance(self.right, PrunedTree):
                self.right.fit_leaf(X[r], y[r], weight[r])
            else:
                self.right.fit(X[r], y[r], weight[r])
