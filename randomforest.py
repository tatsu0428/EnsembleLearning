import numpy as np
import pandas as pd
import support
import random
import entropy
from zeror import ZeroRule
from linear import Linear
from pruning import PrunedTree
from bagging import Bagging

class RandomTree(PrunedTree):
	def __init__(self, features=5, max_depth=5, metric=entropy.gini, leaf=ZeroRule, depth=1):
		super().__init__(max_depth=max_depth, metric=metric, leaf=leaf, depth=depth)
		self.features = features

	def split_tree(self, X, y):
		# 説明変数内の次元から,ランダムに使用する次元を選択する
		index = random.sample(range(X.shape[1]), min(self.features, X.shape[1]))
		# 説明変数内の選択された次元のみ使用して分割
		result = self.split_tree_fast(X[:,index], y)
		# 分割の次元を，元の次元に戻す
		self.feat_index = index[self.feat_index]
		return result

	def get_node(self):
		# 新しくノードを作成する
		return RandomTree(features=self.features, max_depth=self.max_depth, metric=self.metric, leaf=self.leaf, depth=self.depth + 1)


if __name__ == '__main__':
	random.seed( 1 )
	ps = support.get_base_args()
	ps.add_argument( '--trees', '-t', type=int, default=5, help='Num of Trees' )
	ps.add_argument( '--ratio', '-p', type=float, default=1.0, help='Bagging size' )
	ps.add_argument( '--features', '-f', type=int, default=5, help='Num of Features' )
	ps.add_argument( '--depth', '-d', type=int, default=5, help='Max Tree Depth' )
	args = ps.parse_args()

	df = pd.read_csv( args.input, sep=args.separator, header=args.header, index_col=args.indexcol )
	x = df[ df.columns[ :-1 ] ].values

	if not args.regression:
		y, clz = support.clz_to_prob( df[ df.columns[ -1 ] ] )
		plf = Bagging( n_trees=args.trees, ratio=args.ratio,
			tree=RandomTree,
			tree_params={ 'features':args.features, 'max_depth':args.depth, 'metric':entropy.gini, 'leaf':ZeroRule } )
		support.report_classifier( plf, x, y, clz, args.crossvalidate )
	else:
		y = df[ df.columns[ -1 ] ].values.reshape( ( -1, 1 ) )
		plf = Bagging( n_trees=args.trees, ratio=args.ratio,
			tree=RandomTree,
			tree_params={ 'features':args.features, 'max_depth':args.depth, 'metric':entropy.deviation, 'leaf':Linear } )
		support.report_regressor( plf, x, y, args.crossvalidate )

