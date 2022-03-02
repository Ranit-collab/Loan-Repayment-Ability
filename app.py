import numpy as np
import pandas as pd 
from flask import Flask, request,render_template
import pickle

app = Flask(__name__)
#initialize the flask App
model=pickle.load(open('loan_model.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))     

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    int_features=[x for x in request.form.values()]
    final=np.array(int_features)
    df=pd.DataFrame([final])
    scaled=scaler.fit_transform(df)
    data=pd.DataFrame(scaled,columns=['funded_amnt_inv','int_rate','emp_length','annual_inc','pymnt_plan','dti','deling_2yrs',
    'fico_range_low','ing_last_6mths','open_acc','pub_rec','revol_bal','revol_util','total_acc','initial_list_status','total_rec_late_fee',
    'last_pymnt_amnt','last_fico_range_high','last_fico_range_low','collections_12_mths_ex_med','policy_code','application_type','acc_now_dealing',
    'tot_coll_amt','tot_cur_bal','total_bal_il','max_bal_bc','issue_d_month','earliest_cr_line_month','earliest_cr_line_year',
    'last_pymnt_d_year','last_credit_pull_d_month','last_credit_pull_d_year'])
    prediction=model.predict(data)
    output=prediction
    
    if output==1:
        label='A'
    elif output==2:
        label='B'
    elif output==3:
        label='C'
    elif output==4:
        label='D'
    elif output==5:
        label='E'
    elif output==6:
        label='F'
    else:
        label='G'
    
    return render_template('index.html',prediction_text='Loan Grade is {}'.format(label))

if __name__=="__main__":
    pd.set_option('display.max_columns',None)
    app.run(debug=True)
    