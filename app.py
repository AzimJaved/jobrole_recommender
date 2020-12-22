from flask import Flask, request, render_template, redirect, url_for, jsonify
import pickle
import numpy as np
import pandas as pd
import itertools
from model import lst_dct

app = Flask(__name__)

data = pd.read_csv('final_dataset_prod.csv')

forest=pickle.load(open('model.pkl','rb'))


@app.route('/')
def homepage():
    return render_template("homepage.html")


@app.route('/recommend',methods = ['POST', 'GET'])
def recommend():
    if (request.method == 'POST'):
        ind_ = request.form['industry']
        f_area_ = request.form['functionalArea']
        sk1_ = request.form['skill1']
        sk2_ = request.form['skill2']
        sk3_ = request.form['skill3']
        sk4_ = request.form['skill4']
        sk5_ = request.form['skill5']

        f_area = lst_dct[1][f_area_]
        ind = lst_dct[2][ind_]
        sk1 = lst_dct[3][sk1_]
        sk2 = lst_dct[4][sk2_]
        sk3 = lst_dct[5][sk3_]
        sk4 = lst_dct[6][sk4_]
        sk5 = lst_dct[7][sk5_]
        predicted_rolecat = []
        data_j=data[((data['Functional Area'] == f_area_) & (data['Industry'] == ind_)) & ((data['Skill1']==sk1_) | (data['Skill2']==sk2_) | (data['Skill3']==sk3_) | (data['Skill4']==sk4_) | (data['Skill5']==sk5_))]
        if len(data_j) > 0:
            p=list(set(list(data_j['Role Category'].values)))
            predicted_rolecat.extend(p)
        
        test = ['Functional Area_encoded', 'Industry_encoded','Skill1_encoded','Skill2_encoded','Skill3_encoded','Skill4_encoded','Skill5_encoded']
        param = [f_area,ind,sk1,sk2,sk3,sk4,sk5]
        data_test = {}
        i = 0
        for col in test:
            data_test[col] = [param[i]]
            i = i+1
        test_df = pd.DataFrame(data_test)

        predict_code = forest.predict(test_df)   
        predict_code = predict_code.tolist()

        for code in predict_code:
            if code is None:
                print("No role predicted")
                break
            else:
                for key,value in lst_dct[0].items():
                    if value == code:
                        predicted_rolecat.append(key)
        fin = []
        for rc in predicted_rolecat:
            if rc in list(set(list(data['Role Category'].values))):
                fin.append(rc)
            else:
                pass
        predicted_rolecat = list(set(fin))

        final = []
        ready = []
        sk_inp = [sk1_,sk2_,sk3_,sk4_,sk5_]
        for rol_cat in predicted_rolecat:
            data2 = data[data['Role Category'] == rol_cat] #and (data['Functional Area'] == f_area) and (data['Industry'] == ind)]
            role_lt = []
            sk_dct = {}
            intermed = []
            d1 = []
            for role in list(data2['Role'].values):
                if role not in role_lt:
                    role_lt.append(role)
                    intermed.append([rol_cat, role, 5])
            for role in role_lt :
                    data3 = data2[data['Role'] == role]
                    sc = ['Skill1','Skill2','Skill3','Skill4','Skill5']
                    for c in sc:
                        for skill in list(data3[c].values):
                            skill = skill.lower()
                            sk_dct[skill] = sk_dct.get(skill, 0) + 1
                    b = sorted(sk_dct, key = sk_dct.__getitem__, reverse = True)
                    d = readiness(b, rol_cat, role, sk_inp)
                    d1.append(d)
                    a = b[0:12]
                    for sk in a:
                        intermed.append([role, sk, 5])
            ready.append(d1)
            final.append(intermed)
        return render_template('results.html', sen = final, tab = ready, itertools=itertools, jsonify=jsonify)
        # return jsonify({'sen': final, 'tab': ready})
    
    
def readiness(x, cat, role, sk_inp):
    r = 0
    for i in sk_inp:
        if i in x:
            r = r + 1
    if r >= 4:
        t = [role, True, False, False]
        return t

    elif r >= 2 :
        t = [role, False, True, False]
        return t

    else :
        t = [role, False, False, True]
        return t
    


if __name__ == '__main__':
    app.run(debug=True)
