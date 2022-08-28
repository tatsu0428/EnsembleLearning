import numpy as np
import pandas as pd
import support

class Linear:

    def __init__(self, epochs=20, lr=0.01, earlystop=None):

        self.epochs = epochs
        self.lr = lr
        self.earlystop = earlystop
        self.beta = None
        self.norm = None

    def fitnorm(self, X, y):
        # 学習の前にデータに含まれる値の範囲を0から1に正規化するので，
        # そのためのパラメータを保存しておく
        self.norm = np.zeros((X.shape[1]+1, 2))
        self.norm[0, 0] = np.min(y)         # 目的変数の最小値
        self.norm[0, 1] = np.max(y)         # 目的変数の最大値
        self.norm[1:,0] = np.min(X, axis=0) # 説明変数の最小値
        self.norm[1:,1] = np.max(X, axis=0) # 説明変数の最大値

    def normalize(self, X, y=None):
        # データに含まれる値の範囲を0から1に正規化する
        l = self.norm[1:,1] - self.norm[1:,0]
        l[l==0] = 1
        p = (X - self.norm[1:,0]) / l
        q = y
        if y is not None and not self.norm[0,1] == self.norm[0,0]:
            q = (y - self.norm[0,0]) / (self.norm[0,1] - self.norm[0, 0])
        return p, q

    def r2(self, y, z):
        # EarlyStopping用にR2スコアを計算する
        y = y.reshape((-1,))
        z = z.reshape((-1,))
        mn = ((y - z) ** 2).sum(axis=0)
        dn = ((y - y.mean())**2).sum(axis=0)
        if dn == 0:
            return np.inf
        return 1.0 - (mn / dn)

    def fit(self, X, y):
        # 勾配降下法による線形回帰係数の推定を行う
        # 最初にデータに含まれる値の範囲を0から1に正規化する
        self.fitnorm(X, y)
        X, y = self.normalize(X, y)

        # 線形回帰係数
        # 配列の最初の値がy=ax+bのbに，続く値がaになる
        self.beta = np.zeros((X.shape[1]+1,))

        # 繰り返し学習を行う
        for _ in range(self.epochs):
            # 1エポック内でデータを繰り返す
            for p, q in zip(X, y):
                # 現在のモデルによる出力から勾配を求める
                z = self.predict(p.reshape((1, -1)), normalized=True)
                z = z.reshape((1,))
                err = (z - q) * self.lr
                delta = p * err
                # モデルを更新する
                self.beta[0] -= err
                self.beta[1:] -= delta
            # EarlyStoppingが有効ならば
            if self.earlystop is not None:
                # スコアを求めて一定値以上ならば終了
                z = self.predict(X, normalized=True)
                s = self.r2(y, z)
                if self.earlystop <= s:
                    break
        return self

    def predict(self, X, normalized=False):
        # 線形回帰モデルを実行する
        # まずは値の範囲を0から1に正規化する
        if not normalized:
            X, _ = self.normalize(X)
        # 結果を計算する
        z = np.zeros((X.shape[0], 1)) + self.beta[0]
        for i in range(X.shape[1]):
            c = X[:,i] * self.beta[i+1]
            z += c.reshape((-1, 1))
        # 0から1に正規化した値を戻す
        if not normalized:
            z = z * (self.norm[0,1] - self.norm[0,0]) + self.norm[0,0]
        return z

    def __str__(self):
        # モデルの内容を文字列にする
        if type(self.beta) is not type(None):
            s = ["%f"%self.beta[0]]
            e = ["+feat[%d]*%f"%(i+1, j) for i,j in enumerate(self.beta[1:])]
            s.extend(e)
            return "".join(s)
        else:
            return "0.0"


if __name__ == "__main__":

    ps = support.get_base_args()
    ps.add_argument("--epochs", "-p", type=int, default=20, help="Num of Epochs")
    ps.add_argument("--learningrate", "-l", type=float, default=0.01, help="Learning Rate")
    ps.add_argument("--earlystop", "-a", action='store_true', help="Early Stopping")
    ps.add_argument("--stopingvalue", "-v", type=float, default=0.01, help="Early Stopping")
    args = ps.parse_args()

    df = pd.read_csv(args.input, sep=args.separator, header=args.header, index_col=args.indexcol)
    X = df[df.columns[:-1]].values

    if not args.regression:
        print("Not Support")
    else:
        y = df[df.columns[-1]].values.reshape((-1,1))
        if args.earlystop:
            plf = Linear(epochs=args.epochs, lr=args.learningrate, earlystop=args.stopingvalue)
        else:
            plf = Linear(epochs=args.epochs, lr=args.learningrate)
        support.report_regressor(plf, X, y, args.crossvalidate)
