# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 20:43:50 2018

@author: DELL
"""

import pandas as pd
import numpy as np
Buyers_original = pd.read_csv('final_train_table.csv')
#查看是否有缺失值
#print(np.isnan(Buyers_original).any())
#删除缺失值
Buyers_original.dropna(inplace=True)
#查看是否有缺失值
#print(np.isnan(Buyers_original).any())
#更改数据类型
Buyers_original['gender']= Buyers_original['gender'].astype('int')
Buyers_original['gender']= Buyers_original['gender'].astype('category')
Buyers_original['age_range']= Buyers_original['age_range'].astype('int')
Buyers_original['age_range']= Buyers_original['age_range'].astype('category')
Buyers_original['label']= Buyers_original['label'].astype('int')
Buyers_original['label']= Buyers_original['label'].astype('category')

Buyers_original.to_csv('training_data.csv',index = False)

#data
Buyers_X = Buyers_original.drop(['user_id','merchant_id','label'],axis = 1)
#target value
Buyers_y = Buyers_original['label']


from sklearn.linear_model.logistic import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Buyers_X,Buyers_y,random_state = 0)
classifier = MLPClassifier(solver='adam', alpha=1e-5, activation='tanh',
                    hidden_layer_sizes=(10,8,6,4,2), random_state=1)
classifier.fit(X_train,y_train)

'''
Buyers_predicte= pd.read_csv('final_test_table.csv')

Buyers_predicte['gender']= Buyers_predicte['gender'].astype('int')
Buyers_predicte['gender']= Buyers_predicte['gender'].astype('category')
Buyers_predicte['age_range']= Buyers_predicte['age_range'].astype('int')
Buyers_predicte['age_range']= Buyers_predicte['age_range'].astype('category')
Buyers_predicte = Buyers_predicte.drop(['user_id','merchant_id'],axis = 1)
'''


#利用测试数据对模型进行评估
print("Test set score:{:.2f}".format(classifier.score(X_test,y_test)))
#print(classifier.predict_proba(X_test))

'''
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(y_test, classifier.decision_function(X_test))
plt.plot(fpr, tpr, label = 'ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR(Recall)')
plt.title('Evalation',fontsize = 15)


#列表，包含对应顺序排列的所有可能的阈值，对应的的准确率和召回率
from sklearn.metrics import precision_recall_curve
precision, recall ,threholds = precision_recall_curve(y_test,classifier.decision_function(X_test))
plt.plot(precision,recall,label = 'precision recall curve')
plt.xlabel('Precision')
plt.ylabel('Recall')

plt.legend(loc = 1)
plt.show()
'''




