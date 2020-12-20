
import numpy as np 
import pandas as pd
import pickle

data=pd.read_csv('final_dataset_test.csv')

stringcols = ('Role Category','Functional Area','Industry','Skill1','Skill2','Skill3','Skill4','Skill5')
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
X = data[['Functional Area_encoded', 'Industry_encoded','Skill1_encoded','Skill2_encoded','Skill3_encoded','Skill4_encoded','Skill5_encoded']]

# split the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.22, random_state =42)

from sklearn.metrics import classification_report ,f1_score,accuracy_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier

forest=RandomForestClassifier()
forest.fit(X_train,y_train)

pickle.dump(forest,open('model.pkl','wb'), protocol = 4)


