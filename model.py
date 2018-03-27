from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xg
from sklearn.metrics import classification_report,accuracy_score
import numpy as np
import pandas as pd

svc_params = [
	{
		'C': [1, 3, 5, 7, 9],
		'gamma': [ 0.001, 0.01, 0.1, 1, 10],
		'kernel': ['rbf']
	},
	{
		'C': [1, 3, 5, 7, 9],
		'kernel': ['linear']
	}
]

logistic_params = [
	{
		'penalty':['l1','l2'],  
		'C':[0.01,0.05,0.1,0.5,1,5,10,50,100],  
		'solver':['liblinear'],  
		'multi_class':['ovr']
	},  
	{
		'penalty':['l2'],  
		'C':[0.01,0.05,0.1,0.5,1,5,10,50,100],  
		'solver':['lbfgs'],  
		'multi_class':['ovr','multinomial']
	}
]
random_forest_param ={
	'criterion':['gini','entropy'],
	'max_depth':[5,6,7],
	'max_features':['auto',None],
	'min_samples_split':[4,5,6,7]
}
xgboost_param = {
	'max_depth':[4,5,6,7],
	# 'learning_rate':[0.1,0.01],
	# 'objective':['binary:logistic'],
	# 'booster':['gbtree', 'gblinear', 'dart'],
	# 'gamma':[0,0.001],
	'reg_alpha':np.linspace(0.1,1,5),
	'reg_lambda':np.linspace(0.1,1.2,5),
}

models =[
	('Xgboost',xg.XGBClassifier(),xgboost_param),
	('RandomForest', RandomForestClassifier(n_estimators=500),random_forest_param),
	('Logistic' , LogisticRegression(),logistic_params),
	('SVM' , SVC() , svc_params),
]

def model_selection(x_train ,y_train,x_test,y_test):
	for m in models:
		print("evaluting " , m[0])
		clf =GridSearchCV(m[1],m[2],n_jobs=-1,cv=10,return_train_score=True)
		clf.fit(x_train,y_train)
		# result = pd.DataFrame.from_dict(clf.cv_results_)
		# with open(m[0]+'.csv','w') as f:
		# 	result.to_csv(f)
		print('The parameters of the best '+m[0]+' are: ')
		print(clf.best_params_)
		y_pred = clf.predict(x_train)
		print(classification_report(y_true=y_train, y_pred=y_pred))
		y_test_pred = clf.predict(x_test)
		# print(classification_report(y_true=y_test, y_pred=y_test_pred))
		# df_test_y = pd.DataFrame(y_test_pred , columns=['Survived'])
		df = pd.DataFrame(data.get_test_PassengerId()).join(pd.DataFrame(y_test_pred , columns=['Survived']))
		print(df.head())
		df.to_csv('./titanic_test_result_'+m[0]+'.csv',index=False)

import data,preprocess
if '__main__' == __name__:
	train_data = data.get_train_data()
	train_data =preprocess.fill_missing_data(train_data ,sex_cat=True, embarked_one_hot=True)
	# train_data = preprocess.feature_selection(train_data)
	# train_data = preprocess.detect_outlier(train_data,drop=True)
	print(train_data.head())
	x_train,y_train = data.split(train_data)
	# print(y_train.values)
	x_test,y_test = data.get_test_x(),data.get_test_y()
	x_test =preprocess.fill_missing_data(x_test,is_train=False,sex_cat=True, embarked_one_hot=True)
	# poly = PolynomialFeatures(2,interaction_only=True)
	# x_train = poly.fit_transform(x_train.values)
	# x_test = poly.fit_transform(x_test.values)
	model_selection(x_train.values,y_train.values,x_test.values,y_test.values)
	