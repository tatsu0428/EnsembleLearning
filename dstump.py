import numpy as np
import pandas as pd
import support
import entropy
from zeror import ZeroRule
from linear import Linear

class DecisionStump:
    def __init__(self, metric=entropy.gini, leaf=ZeroRule):
        self.metric = metric
        self.leaf = leaf
        self.left = None
        self.right = None
        self.feat_index = 0
        self.feat_val = np.nan
        self.score = np.nan

    def make_split(self, feat, val):
        # featをval以下と以上で分割するインデックスを返す
        left, right = [], []
        for i, v in enumerate(feat):
            if v < val:
                left.append(i)
            else:
                right.append(i)
        return left, right

    def make_loss(self, y1, y2, l, r):
        # yをy1とy2で分割したときのMetrics関数の重み付き合計を返す
        if y1.shape[0] == 0 or y2.shape[0] == 0:
            return np.inf
        total = y1.shape[0] + y2.shape[0]
        m1 = self.metric(y1) * (y1.shape[0]/total)
        m2 = self.metric(y2) * (y2.shape[0]/total)
        return m1 + m2

    def split_tree(self, X, y):
        # データを分割して左右の枝に属するインデックスを返す
        self.feat_index = 0
        self.feat_val = np.inf
        score = np.inf
        # 左右のインデックス
        left, right = list(range(X.shape[0])), []
        # 説明変数内の全ての次元に対して
        for i in range(X.shape[1]):
            feat = X[:,i] # その次元の値の配列
            for val in feat:
                # 最もよく分割する値を返す
                l, r = self.make_split(feat, val)
                loss = self.make_loss(y[l], y[r], l, r)
                if score > loss:
                    score = loss
                    left = l
                    right = r
                    self.feat_index = i
                    self.feat_val = val
        self.score = score # 最良の分割点のスコア
        return left, right

    def fit(self, X, y):
        # 左右の葉を作成する
        self.left = self.leaf()
        self.right = self.leaf()
        # データを左右の葉に振り分ける
        left, right = self.split_tree(X, y)
        # 左右の葉を学習させる
        if len(left) > 0:
            self.left.fit(X[left], y[left])
        if len(right) > 0:
            self.right.fit(X[right], y[right])
        return self

    def predict(self, X):
        # 説明変数から分割した左右のインデックスを取得
        feat = X[:, self.feat_index]
        val = self.feat_val
        l, r = self.make_split(feat, val)
        # 左右の葉を実行して結果を作成する
        z = None
        if len(l) > 0 and len(r) > 0:
            left = self.left.predict(X[l])
            right = self.right.predict(X[r])
            z = np.zeros((X.shape[0], left.shape[1]))
            z[l] = left
            z[r] = right
        elif len(l) > 0:
            z = self.left.predict(X)
        elif len(r) > 0:
            z = self.right.predict(X)
        return z

    def __str__(self):
        return "\n".join([
            "  if feat[ %d ] <= %f then:"%( self.feat_index, self.feat_val ),
			"    %s"%( self.left, ),
			"  else",
			"    %s"%( self.right, )
        ])


if __name__ == "__main__":
    ps = support.get_base_args()
    ps.add_argument("--metric", "-m", default="", help="Metric function")
    ps.add_argument("--leaf", "-l", default="", help="Leaf class")
    args = ps.parse_args()

    df = pd.read_csv(args.input, sep=args.separator, header=args.header, index_col=args.indexcol)
    X = df[df.columns[:-1]].values

    if args.metric == "div":
        mt = entropy.deviation
    elif args.metric == "infgain":
        mt = entropy.infgain
    elif args.metric == "gini":
        mt = entropy.gini
    else:
        mt = None

    if args.leaf == "zeror":
        lf = ZeroRule
    elif args.leaf == "linear":
        lf = Linear
    else:
        lf = None

    if not args.regression:
        y, clz = support.clz_to_prob(df[df.columns[-1]])
        if mt is None:
            mt = entropy.gini
        if lf is None:
            lf = ZeroRule
        plf = DecisionStump(metric=mt, leaf=lf)
        support.report_classifier(plf, X, y, clz, args.crossvalidate)
    else:
        y = df[df.columns[-1]].values.reshape((-1, 1))
        if mt is None:
            mt = entropy.deviation
        if lf is None:
            lf = Linear
        plf = DecisionStump(metric=mt, leaf=lf)
        support.report_regressor(plf, X, y, args.crossvalidate)
