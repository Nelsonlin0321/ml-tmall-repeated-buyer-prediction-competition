# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 23:00:49 2018

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


#data
Buyers_X = Buyers_original.drop(['user_id','merchant_id','label'],axis = 1)
#target value
Buyers_y = Buyers_original['label']


from sklearn.linear_model.logistic import LogisticRegression
classifier=LogisticRegression()
classifier.fit(Buyers_X,Buyers_y)





Buyers_predict= pd.read_csv('final_test_table.csv')
# 查看是否有缺失值
print(np.isnan(Buyers_predict).any())
Buyers_predict.fillna(Buyers_predict.mean(),inplace = True)

print(np.isnan(Buyers_predict).any())
print(Buyers_predict.dtypes)


Buyers_predict['gender']= Buyers_predict['gender'].astype('int')
Buyers_predict['gender']= Buyers_predict['gender'].astype('category')
Buyers_predict['age_range']= Buyers_predict['age_range'].astype('int')
Buyers_predict['age_range']= Buyers_predict['age_range'].astype('category')

X_predict = Buyers_predict.drop(['user_id','merchant_id'],axis = 1)


df_Prob = pd.DataFrame(classifier.predict_proba(X_predict))
df_prob_1 = df_Prob[1]

df_result1 = Buyers_predict[['user_id','merchant_id']]

df_result = pd.concat([df_result1,df_prob_1],axis= 1,ignore_index = True)

print(df_result)
#df_result1.columns = ['user_id','merchant_id','Prob']

df_result.to_csv('result.csv',index = False)
#df_result = df
#temp_list = classifier.predict_proba(X_predict).tolist()






