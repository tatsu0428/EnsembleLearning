import numpy as np
import support
import entropy
from zeror import ZeroRule
from linear import Linear
from dtree import DecisionTree

def reducederror(node, X, y):
    # ノードが葉でなかったら
    if isinstance(node, PrunedTree):
        # 左右の分割を得る
        feat = X[:,node.feat_index]
        val = node.feat_val
        l, r = node.max_split(feat, val)
        # 左右にデータが振り分けられるか
        if val is np.inf or len(r) == 0:
            return reducederror(node.left, X, y)  # 1つの枝のみの場合，その枝で置き換える
        elif len(l) == 0:
            return reducederror(node.right, X, y) # 1つの枝のみの場合，その枝で置き換える
        # 左右の枝を更新する
        node.left = reducederror(node.left, X[l], y[l])
        node.right = reducederror(node.right, X[r], y[r])
        # 学習データに対するスコアを計算する
        p1 = node.predict(X)
        p2 = node.left.predict(X)
        p3 = node.right.predict(X)
        # クラス分類かどうか
        if y.shape[1] > 1:
            # 誤分類の個数をスコアにする
            ya = y.argmax(axis=1)
            d1 = np.sum(p1.argmax(axis=1) != ya)
            d2 = np.sum(p2.argmax(axis=1) != ya)
            d3 = np.sum(p3.argmax(axis=1) != ya)
        else:
            # 二乗平均誤差をスコアにする
            d1 = np.mean((p1-y)**2)
            d2 = np.mean((p2-y)**2)
            d3 = np.mean((p3-y)**2)
        if d2 <= d1 or d3 <= d1: # 左右の枝どちらかだけでスコアが悪化しない
            # スコアの良い方の枝を返す
            if d2 < d3:
                return node.left
            else:
                return node.right
    # 現在のノードを返す
    return node
