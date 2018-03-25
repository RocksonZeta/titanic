import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import preprocess

mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus']=False
sns.set(style="white", color_codes=True)


def brief(df):
	print("Head:")
	print(df.head())
	print("Describe:")
	print(df.describe())
	print("Category:")
	cats = ['Survived','Pclass' ,'Sex','SibSp','Parch']
	for cat in cats:
		print(cat , df[cat].unique())
	print("Na detect:")
	for col in df.columns:
		na = df.loc[df[col].isnull() , col]
		if len(na) >0 :
			print(col," NA:",len(na),"/",len(df))





def plot(df):
	plt.figure(num=1)
	sns.jointplot(x="Age", y="Fare", data=df, size=5)
	plt.figure(num=2)
	df.plot(kind="scatter", x="Age", y="Fare")
	plt.figure(num=3)
	sns.FacetGrid(df, hue="Survived", size=5).map(plt.scatter, "Age", "Parch").add_legend()
	plt.figure(num=4)
	plt.subplot(211)
	sns.boxplot(x="Survived", y="Age", data=df)
	plt.subplot(212)
	sns.boxplot(x="Survived", y="Parch", data=df)
	plt.figure(num=5)	
	sns.violinplot(x="Survived", y="Age", data=df, size=6)  
	# plt.figure(num=6)	
	# sns.FacetGrid(df, hue="Survived", size=6).map(sns.kdeplot, "Age").add_legend()
	plt.figure(num=7)
	sns.pairplot(df[['Age','Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']], hue="Survived", size=2)
	# plt.figure(num=8)
	# sns.pairplot(df[['Age','Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']], hue="Survived", diag_kind="kde",size=2)  
	

def dim_reduction_plot(df):
	df = df.sample(500, random_state =1)
	# print("PCA dim reduction plot")	
	plt.figure("PCA dim reduction plot" , figsize=(8,6) )
	# cm = mpl.colors.ListedColormap(['r','g'])
	pca = PCA(n_components=2)
	c = df.Survived.values
	data = pca.fit_transform(df)
	pdata = pd.DataFrame(data , index = df.index)
	d = pdata[df['Survived']==0]
	p0 = plt.scatter(d[0],d[1] , c='r' , marker='x',linewidths=1)
	d = pdata[df['Survived']==1]
	p1 = plt.scatter(d[0],d[1] , c='' , marker='o', edgecolors='g',linewidths=1)
	plt.legend([p0, p1], ['Survived=0', 'Survived=1'], loc='upper right', scatterpoints=1) 
	
	# print("TSNE dim reduction plot")
	plt.figure("TSNE dim reduction plot",figsize=(8,6))
	tsne = TSNE()
	tsne.fit_transform(df)
	tdata = pd.DataFrame(tsne.embedding_,index=df.index)
	d = tdata[df['Survived']==0]
	p0 = plt.scatter(d[0],d[1] , c='r' , marker='x',linewidths=1)
	d = tdata[df['Survived']==1]
	p1 = plt.scatter(d[0],d[1] , c='' , marker='o', edgecolors='g',linewidths=1)
	plt.legend([p0, p1], ['Survived=0', 'Survived=1'], loc='upper right', scatterpoints=1) 
	

if '__main__' == __name__:
	df = pd.read_csv('data/train.csv')
	# brief(df)
	df = preprocess.fill_missing_data(df)
	# brief(df)
	# plot(df)
	dim_reduction_plot(df)
	plt.show()
