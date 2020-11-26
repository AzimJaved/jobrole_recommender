
import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


data=pd.read_csv('final_dataset.csv')


for col in ['skill1', 'skill2', 'skill3', 'skill4']:
    n= []
    x = list(data[col].values)
    for item in x:
        item = item.strip().lstrip()
        n.append(item)
    data[col.capitalize()] = n

data.drop(columns = ['skill1', 'skill2', 'skill3', 'skill4', 'Unnamed: 0', 'Key Skills'], axis = 1, inplace=True)
data.rename(columns={'Role':'Job Role','Role Category':'Role'},inplace=True)
data.rename(columns={'Job Role':'Role Category'},inplace=True)
data = data.drop(index=[1338,11854,13166,29456])

sk_col = ['Skill1', 'Skill2', 'Skill3', 'Skill4']
s1 = list(data['Skill1'].values)
s2 = list(data['Skill2'].values)
s3 = list(data['Skill3'].values)
s4 = list(data['Skill4'].values)
m = [s1,s2,s3,s4]
for col in m:
    for entry in col:
        entry = entry.lower()


for k in range(4):
    for i in m[k]:
        for j in range(k+1,4):
            if i in m[j]:
                ind = m[j].index(i)
                m[k][ind], m[j][ind] = m[j][ind], m[k][ind]
                
data.drop(sk_col, axis=1, inplace=True)
data['Skill1'] = m[0]
data['Skill2'] = m[1]
data['Skill3'] = m[2]
data['Skill4'] = m[3]

stringcols = ('Role Category','Functional Area','Industry','Skill1','Skill2','Skill3','Skill4')
from sklearn.preprocessing import LabelEncoder

lst_dct = []
dct = {}
for c in stringcols:
    lbl = LabelEncoder() 
    lbl.fit(list(data[c].values)) 
    data[c+'_encoded'] = lbl.transform(list(data[c].values))
    i = iter(data[c].values) 
    j = iter(data[c+'_encoded'].values)
    while True :
        try:
            dct[next(i)] = int(next(j))
        except:
            break
    lst_dct.append(dct)

y = data['Role Category_encoded']
X = data[['Functional Area_encoded', 'Industry_encoded','Skill1_encoded','Skill2_encoded','Skill3_encoded','Skill4_encoded']]
# split the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.22, random_state =42)
from sklearn.metrics import classification_report ,f1_score,accuracy_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier()
forest.fit(X_train,y_train)

pickle.dump(forest,open('model.pkl','wb'))


