import numpy as np
import pandas as pd
import support

class ZeroRule:

    def __init__(self):
        self.r = None

    def fit(self, X, y):
        self.r = np.mean(y, axis=0)
        return self

    def predict(self, X):
        z = np.zeros((len(X), self.r.shape[0]))
        return z + self.r

    def __str__(self):
        return str(self.r)


if __name__ == "__main__":

    ps = support.get_base_args()
    args = ps.parse_args()

    df = pd.read_csv(args.input, sep=args.separator, header=args.header, index_col=args.indexcol)
    X = df[df.columns[:-1]].values

    if not args.regression:
        y, clz = support.clz_to_prob(df[df.columns[-1]])
        plf = ZeroRule()
        support.report_classifier(plf, X, y, clz, args.crossvalidate)
    else:
        y = df[df.columns[-1]].values.reshape((-1, 1))
        plf = ZeroRule()
        support.report_regressor(plf, X, y, args.crossvalidate)
