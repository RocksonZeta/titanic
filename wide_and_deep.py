import tensorflow as tf
import data
import matplotlib.pyplot as plt
import pandas as pd

_CSV_COLUMNS = [
	'PassengerId','Survived','Pclass','Name','Sex','Age',
	'SibSp','Parch','Ticket','Fare','Cabin','Embarked'
]
_CSV_COLUMN_DEFAULTS = [[0], [0],[0], [''], ['male'], [20.0],
	[0], [0], [''], [40.0],[''],['S']]


numeric_column = tf.feature_column.numeric_column
bucketized_column = tf.feature_column.bucketized_column
categorical_column_with_hash_bucket = tf.feature_column.categorical_column_with_hash_bucket
crossed_column = tf.feature_column.crossed_column
embedding_column = tf.feature_column.embedding_column
indicator_column = tf.feature_column.indicator_column
categorical_column_with_vocabulary_list = tf.feature_column.categorical_column_with_vocabulary_list
pandas_input_fn = tf.estimator.inputs.pandas_input_fn


Fare = numeric_column('Fare')
Pclass = numeric_column('Pclass')
Age = numeric_column('Age')
SibSp = numeric_column('SibSp')
Parch = numeric_column('Parch')

Embarked = categorical_column_with_vocabulary_list(
	'Embarked',['S','C','Q']
)
Sex = categorical_column_with_vocabulary_list(
	'Sex',['female','male']
)

Age_buckets = bucketized_column(Age,boundaries=[10,18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
Fare_buckets = bucketized_column(Fare,boundaries=[75,110, 125, 150, 200, 250])
Pclass_buckets = bucketized_column(Pclass,boundaries=[0,1,2,3])
SibSp_buckets = bucketized_column(SibSp,boundaries=[0,1,2,3])
Parch_buckets = bucketized_column(Parch,boundaries=[0,1,2,3])

base_columns = [
	Fare,Pclass,Age_buckets,SibSp,Parch,Embarked,Sex
]

crossed_columns=[
	crossed_column([Age_buckets,Sex] , hash_bucket_size=1000),
	crossed_column([Age_buckets,Fare_buckets] , hash_bucket_size=1000),
	crossed_column([Age_buckets,Pclass_buckets] , hash_bucket_size=1000),
	crossed_column([Age_buckets,Embarked] , hash_bucket_size=1000),
	crossed_column([Sex,Embarked] , hash_bucket_size=1000),
	crossed_column([Sex,Pclass_buckets] , hash_bucket_size=1000),
	crossed_column([Sex,Fare_buckets] , hash_bucket_size=1000),
	crossed_column([Sex,SibSp_buckets] , hash_bucket_size=1000),
	crossed_column([Sex,Parch_buckets] , hash_bucket_size=1000),
	crossed_column([Pclass_buckets,Fare_buckets] , hash_bucket_size=1000),
	crossed_column([SibSp_buckets,Parch_buckets] , hash_bucket_size=1000),
	crossed_column([Pclass_buckets,Embarked] , hash_bucket_size=1000),
]
wide_columns = base_columns + crossed_columns

deep_columns = [
	Age, Fare,Pclass,SibSp,Parch,
	indicator_column(Sex),
	indicator_column(Embarked),
]

hidden_units = [100,75, 50, 25]

run_config = tf.estimator.RunConfig().replace(
    session_config=tf.ConfigProto(device_count={'GPU': 0}))

model = tf.estimator.DNNLinearCombinedClassifier(
        model_dir='titanic_wd_model',
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
		# dnn_dropout=0.3,
		dnn_optimizer='Adam',
        config=run_config)

import preprocess
train_data = data.get_train_data()
print(train_data.head())
# train_data = preprocess.detect_outlier(train_data,True)
train_data =preprocess.fill_missing_data(train_data)
x,y = data.split(train_data)
x_test ,y_test = data.get_test_x(),data.get_test_y()
x_test =preprocess.fill_missing_data(x_test,False)


# for n in range(50):
# 	model.train(input_fn=pandas_input_fn(x,y,shuffle=True))
# 	results = model.evaluate(input_fn=pandas_input_fn(x,y,shuffle=False))
# 	# Display evaluation metrics
# 	print('----------- Results at epoch', (n + 1),'-------------')

# 	for key in sorted(results):
# 		print('%s: %s' % (key, results[key]))

import numpy as np
results = model.predict(input_fn=pandas_input_fn(x_test,y_test,shuffle=False))
df = np.stack(pd.DataFrame(results).class_ids.values ,axis=0).reshape(-1)
df = pd.DataFrame(data.get_test_PassengerId()).join(pd.DataFrame(df , columns=['Survived']))
print(df)
df.to_csv('./titanic_test_result.csv',index=False)
# for r in results:
# 	print(r['classes'])
# 	break