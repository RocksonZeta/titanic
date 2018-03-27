import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,IsolationForest
# from sklearn.feature_selection import SelectKBest
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import data




# first col of cols as target(y) , rest is x
def randomForesetSetValue(df,cols , regression=True):
	data = df[cols]
	target = cols[0]
	train_data = data[data[target].notnull()]
	test_data = data[data[target].isnull()]
	x,y =train_data.values[:,1:], train_data.values[:,0]
	if regression == True:
		model = RandomForestRegressor(n_estimators=200)
	else:
		model = RandomForestClassifier(n_estimators=200)
	model.fit(x,y)
	target_predict = model.predict(test_data.values[:,1:])
	df.loc[data[target].isnull(),target] = target_predict

def drop_col(df):
	return df.drop(['PassengerId','Cabin','Name','Ticket'],axis=1)
def fill_missing_data(df , is_train=True ,sex_cat=False, embarked_one_hot=False):
	print("preprocess data")
	# df = drop_col(df)
	if sex_cat:
		df['Sex'] = df.Sex.map({'female':0,'male':1}).astype(int)
	# Fare should > 0, fill Fare by pclass ,using median of pclass
	if len(df.Fare[df.Fare.isnull()]) > 0:
		fare = np.zeros(3)
		for f in range(0, 3):
			fare[f] = df[df.Pclass == f + 1]['Fare'].dropna().median()
		for f in range(0, 3):  # loop 0 to 2
			df.loc[(df.Fare.isnull()) & (df.Pclass == f + 1), 'Fare'] = fare[f]
	#默认用S
	df.loc[(df.Embarked.isnull()), 'Embarked'] = 'S'
	if embarked_one_hot:
		df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2, 'U': 0}).astype(int)
		embarked_data = pd.get_dummies(df.Embarked)
		embarked_data = embarked_data.rename(columns=lambda x: 'Embarked_' + str(x))
		df = pd.concat([df, embarked_data], axis=1)
		df = df.drop('Embarked',axis=1)

	if is_train:
		randomForesetSetValue(df , ['Age','Survived', 'Fare', 'Parch', 'SibSp', 'Pclass'])
	else :
		randomForesetSetValue(df , ['Age', 'Fare', 'Parch', 'SibSp', 'Pclass'])
	return df

def feature_importance(df):
	import xgboost as xg
	
	x = df.drop('Survived',axis=1)
	y = df.Survived
	rf = RandomForestClassifier(n_estimators=200)
	rf.fit(x,y)
	features = x.columns
	feature_imp = sorted(zip(features,rf.feature_importances_) ,key=lambda x : x[1] , reverse=True)
	print("RandomForestClassifier feature importances:")
	print(feature_imp)

	data_train = xg.DMatrix(x, label=y)
	# watch_list = [(data_test, 'eval'), (data_train, 'train')]
	param = {'max_depth': 6, 'eta': 0.8, 'silent': 1, 'objective': 'binary:logistic'}
			#'subsample': 1, 'alpha': 0, 'lambda': 0, 'min_child_weight': 1}
	bst = xg.train(param, data_train, num_boost_round=100)
	xg.plot_importance(bst)


#Isolation Forest,Local Outlier Factor,DBSCAN
def detect_outlier(df , drop=False):
	# max_samples=400
	# df = df.sample(max_samples)
	# x = df.drop('Survived',axis=1)
	# y = df.Survived
	clf = IsolationForest(n_estimators=200)
	clf.fit(df)
	result = clf.predict(df)
	result_df = pd.Series(result)
	if drop :
		df = df.drop(result_df[result_df==-1].index)
	return df

def feature_selection(df):
	df = df.drop(['Embarked_S','Embarked_C','Embarked_Q'] , axis=1)

	return df

import preview
import sys
if '__main__' == __name__:
	print(sys.path)
	df = data.get_train_data()
	df = fill_missing_data(df)
	# df = feature_selection(df)
	feature_importance(df)
	preview.dim_reduction_plot(df)
	plt.show()
	print(df.head())
	# print(len(df))
	# df = detect_outlier(df,True)
	# print(len(df))