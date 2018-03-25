import pandas as pd

pd.set_option('display.width',400,'precision', 5)

train_file = 'data/train.csv'
test_file = 'data/test.csv'
test_y_file = 'data/gender_submission.csv'
def get_train_data(drop=True):
	df = pd.read_csv(train_file)
	if drop:
		return df.drop(['PassengerId','Ticket','Cabin','Name'],axis=1)
	return df
def get_test_x(drop=True):
	df = pd.read_csv(test_file)
	if drop:
		return df.drop(['PassengerId','Ticket','Cabin','Name'],axis=1)
	return df
def get_test_y():
	return pd.read_csv(test_y_file).Survived

# def get_data():
# 	train = get_train_data()
# 	x_train,y_train = train.drop('Survived',axis=1),train.Survived
# 	x_test = get_test_train()
# 	y_test = get_test_y().Survived

# 	return x_train,x_test,y_train,y_test

def split(df):
	return df.drop('Survived',axis=1),df.Survived