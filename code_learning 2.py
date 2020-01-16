import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn import datasets, svm, metrics,cross_validation, ensemble
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Flatten, Activation

from keras.models import load_model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras import optimizers
from keras.layers.normalization import BatchNormalization
import tensorflow as tf




def random(Xtrain, Ytrain):
    r_list = np.array(range(0,len(Xtrain)))
    np.random.shuffle(r_list)
    Xtrain = Xtrain[r_list]
    Ytrain = Ytrain[r_list]
    return Xtrain,Ytrain

def split(Xdata, Ydata, per):
	split_len = int(len(Xdata)*per)
	return Xdata[:split_len],Xdata[split_len:],Ydata[:split_len],Ydata[split_len:]

def tar_preprocess(Ytrain):
	#t_Ytrain = [Ytrain[i] for i in range(len(Ytrain)) if Ytrain[i]<=1000]
	for i in range(len(Ytrain)):
		#5.5 / 5.75 / 6
		'''
		if(Ytrain[i]<=5.5):
			Ytrain[i]=0
		elif(Ytrain[i]>5.5 and Ytrain[i]<=5.75):
			Ytrain[i]=1
		elif(Ytrain[i]>5.75 and Ytrain[i]<=6):
			Ytrain[i]=2
		elif(Ytrain[i]>6):
			Ytrain[i]=3
		'''
		
		if(Ytrain[i]==0):
			Ytrain[i]=0
		elif(Ytrain[i]>0 and Ytrain[i]<=125):
			Ytrain[i]=1
		elif(Ytrain[i]>125):
			Ytrain[i]=2

	return Ytrain

def one_hot_encode(Xdata):
	num_cat = Xdata.shape[1]
	#=========Label encode=============
	enc = preprocessing.LabelEncoder() #encode with label
	X_label = Xdata[:,0]
	for co in range(1,num_cat):
		X_label = np.concatenate((X_label,Xdata[:,co]),axis=0)
	X_label =X_label.flatten()
	enc.fit(X_label)
	#==================================

	#=========one hot encode===========
	f_encode = []
	for ec in range(0,num_cat):
		f_encode.append(enc.transform(Xdata[:,ec]))
	f_encode = np.array(f_encode)
	f_encode = np.transpose(f_encode)
	one_hot = preprocessing.MultiLabelBinarizer()
	code_encode = one_hot.fit_transform(f_encode)
	#==================================
	print(one_hot.classes_) #印出種類類別（label encoder label 過後的）
	#print(code_encode.shape,code_encode[0])

	return code_encode

def PCA_reduce_dimension(Xdata,dim):

	Xdata = np.array(Xdata).astype('int')
	pca = PCA(n_components = dim)
	pca.fit(Xdata)
	Xdata = pca.transform(Xdata)

	return Xdata

def SVM_train(Xtrain, Ytrain, Xtest, Ytest):

	classifier = svm.SVC(gamma=0.001,verbose=True)
	classifier.fit(Xtrain,Ytrain)
	predicted = classifier.predict(Xtest)

	print(confusion_matrix(Ytest,predicted))
	print('\n')
	print(classification_report(Ytest,predicted))

def RF_train(Xtrain, Ytrain, Xtest, Ytest):
	
	forest = ensemble.RandomForestClassifier(n_estimators = 100,max_features=10)
	forest_fit = forest.fit(Xtrain, Ytrain)

	test_predicted = forest.predict(Xtest)		

	accuracy = metrics.accuracy_score(Ytest, test_predicted)
	
	print(confusion_matrix(Ytest,test_predicted))
	print("accuracy : ",accuracy)
	

def Neural_network(Xtrain, Ytrain, Xtest, Ytest,cat_num):

	Ytrain = to_categorical(Ytrain, cat_num)
	Ytest = to_categorical(Ytest, cat_num)

	model = Sequential()
	model.add(Dense(units = 30, input_dim=Xtrain.shape[1]))
	model.add(Activation("relu"))
	model.add(Dropout(0.3))
	model.add(Dense(units = cat_num))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
	model.fit(Xtrain, Ytrain, epochs=5, batch_size=32)
	loss, accuracy = model.evaluate(Xtest, Ytest)
	print("\n=====Neural_network======\n")
	
	print("accuracy : ",accuracy)
	print("loss : ",loss)
	#print(confusion_matrix(Ytest,test_predicted))

if __name__== "__main__":

	range_name = "point/raw_data_all.csv"
	range_file = pd.read_csv(range_name,header=0,low_memory=False).as_matrix()
	trainX = []
	trainY = []
	algo = sys.argv[1]
	cat_num = 3

	for r in range_file: 
		#Hos_type找不到的就跳過，只有兩筆而已
		if(r[9]=="None"): 
			continue
		#選擇要加入的訓練資料欄位 
		trainX.append([r[0],r[1],r[2],r[5],r[8],r[9]]) #r[5],r[8]

		if(r[3]<=0):
			r[3] = 0
		#選擇要加入的健保點數資料欄位 
		trainY.append(r[17]-r[13])

	trainX = np.array(trainX)
	trainY = np.array(trainY)
	#資料亂數洗牌一遍
	(Xdata,Ydata) = random(trainX,trainY)
	#標上分層的標籤
	Ydata = tar_preprocess(Ydata)

	Xdata = np.array(Xdata)
	Ydata = np.array(Ydata)
	#看分幾層range裡就填多少，因出相對應比例
	for p in range(cat_num):
		print("Ans ",p," percentage : ",len(Ydata[Ydata==p])/len(Ydata))

	#one hot encoding,把要轉換成01的input資料放入
	code_encode = one_hot_encode(Xdata[:,:3])

	Xdata = np.concatenate((Xdata[:,3:],code_encode),axis=1)
	#=============降維==============
	#PCA_reduce_dimension(Xdata,12) #後面數字是要降到的維度
	#==============================
	print(Xdata.shape)
	(Xtrain,Xtest,Ytrain,Ytest)=split(Xdata,Ydata,0.8)

	if(algo == "svm"):
		SVM_train(Xtrain, Ytrain,Xtest,Ytest)
	elif(algo == "rf"):
		RF_train(Xtrain, Ytrain,Xtest,Ytest)
	elif(algo == "nn"):
		Neural_network(Xtrain, Ytrain, Xtest, Ytest,cat_num)
	else:
		print("Algorithm not implemented")

	'''
	#======test plot======
	plt_data = []
	plt_axis = []

	for k in range(0,100,5):
		plt_axis.append(k)
		plt_data.append(np.percentile(Ydata,k))
		print(k," :",np.percentile(Ydata,k))

	plt_data = np.array(plt_data)
	plt_axis = np.array(plt_axis)
	plt.plot(plt_axis,plt_data)
	plt.show()

	'''

