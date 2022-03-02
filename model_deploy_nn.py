import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import LabelBinarizer
from keras.models import load_model

model = load_model('NN_model.h5')
ss = joblib.load('standard_scaler_nn.bin')

def predict_grade(X):
    X = ss.transform(X)
    pred = model.predict(X)
    output = np.argmax(pred)
    return output

def main():
    html_temp = """
    <div style="background-color:orange ;padding=10px">
    <h2 style="color:green,text-align:center;">Loan Grade Prediction</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    funded_amnt_inv = st.number_input('Invested Funded Amount',0.00,10**6.00)
    int_rate = st.slider('Initial rate',5.31,31.0)
    emp_length = st.selectbox('Employee length(in years)',[0,1,2,3,4,5,6,7,8,9,10])
    annual_inc = st.number_input('Annual Income',0.00,10**7.00)
    pymnt_plan = st.selectbox('payment plan',['yes','no'])
    dti = st.number_input('debt-to-income ratio',0.00,1000.00)
    deling_2yrs = st.number_input('No. of times 30+ days payment due in last 2 years',0,100)
    fico_range_low = st.number_input('low FICO score',0.00,1000.00)
    ing_last_6mths = st.number_input('No. of times 30+ days payment due in last 6 months',0,10)
    open_acc = st.number_input('Open Accounts',0,200)
    pub_rec = st.number_input('Public Records',0,100)
    revol_bal = st.number_input('Revolving Balance',0.00,10**9.00)
    revol_util = st.number_input('Revolving Utilization Rate',0.0,1000.0)
    total_acc = st.number_input('Total Accounts',1,200)
    initial_list_status = st.selectbox('initial list status',['Whole','Fractional market'])
    total_rec_late_fee = st.number_input('total late fee',0.00,5000.00)
    last_pymnt_amnt = st.number_input('last payment amount',0.00,10**6.00)
    last_fico_range_high = st.number_input('last high FICO score',0.0,1000.0)
    last_fico_range_low = st.number_input('last low FICO score',0.0,1000.0)
    collections_12_mths_ex_med = st.number_input('collections',0.0,50.0)
    policy_code = st.selectbox('policy code',[1.0])
    application_type = st.selectbox('application type',['Individual','Joint app'])
    acc_now_dealing =st.number_input('account dealing',0,20)
    tot_coll_amt = st.number_input('total collected amount',0.00,10**7.00)
    tot_cur_bal = st.number_input('total current balance',0.00,10**7.00)
    total_bal_il = st.number_input('total bal il',0.00,2*10**6.00)
    max_bal_bc = st.number_input('max bal bc',0.00,2*10**6.00)
    issue_d = st.date_input('issue date')
    earliest_cr_line = st.date_input('earliest credit line')
    last_pymnt_d = st.date_input('last payment date')
    last_credit_pull_d = st.date_input('last credit pull date')

    df = pd.DataFrame({'funded_amnt_inv':funded_amnt_inv,
    'int_rate':int_rate,
    'emp_length':emp_length,
    'annual_inc':annual_inc,
    'pymnt_plan':pymnt_plan,
    'dti':dti,
    'deling_2yrs':deling_2yrs,
    'fico_range_low':fico_range_low,
    'ing_last_6mths':ing_last_6mths,
    'open_acc':open_acc,
    'pub_rec':pub_rec,
    'revol_bal':revol_bal,
    'revol_util':revol_util,
    'total_acc':total_acc,
    'initial_list_status':initial_list_status,
    'total_rec_late_fee':total_rec_late_fee,
    'last_pymnt_amnt':last_pymnt_amnt,
    'last_fico_range_high':last_fico_range_high,
    'last_fico_range_low':last_fico_range_low,
    'collections_12_mths_ex_med':collections_12_mths_ex_med,
    'policy_code':policy_code,
    'application_type':application_type,
    'acc_now_dealing':acc_now_dealing,
    'tot_coll_amt':tot_coll_amt,
    'tot_cur_bal':tot_cur_bal,
    'total_bal_il':total_bal_il,
    'max_bal_bc':max_bal_bc,
    'issue_d':issue_d,
    'earliest_cr_line':earliest_cr_line,
    'last_pymnt_d':last_pymnt_d,
    'last_credit_pull_d':last_credit_pull_d},index=[0],
    columns=['funded_amnt_inv',
     'int_rate',
     'emp_length',
     'annual_inc',
     'pymnt_plan',
     'dti',
     'deling_2yrs',
     'fico_range_low',
     'ing_last_6mths',
     'open_acc',
     'pub_rec',
     'revol_bal',
     'revol_util',
     'total_acc',
     'initial_list_status',
     'total_rec_late_fee',
     'last_pymnt_amnt',
     'last_fico_range_high',
     'last_fico_range_low',
     'collections_12_mths_ex_med',
     'policy_code',
     'application_type',
     'acc_now_dealing',
     'tot_coll_amt',
     'tot_cur_bal',
     'total_bal_il',
     'max_bal_bc','issue_d','earliest_cr_line','last_pymnt_d','last_credit_pull_d'])

    for col in ['issue_d','earliest_cr_line','last_pymnt_d','last_credit_pull_d']:
        date = pd.to_datetime(df[col],format='%Y/%m/%d')
        df[col+'_year'] = date.dt.year
        df[col+'_month'] = date.dt.month

    label_binary_columns = ['pymnt_plan','initial_list_status','application_type']

    labelb = LabelBinarizer()
    for col in label_binary_columns:
        df[col] = labelb.fit_transform(df[col])

    features = ['funded_amnt_inv',
     'int_rate',
     'emp_length',
     'annual_inc',
     'pymnt_plan',
     'dti',
     'deling_2yrs',
     'fico_range_low',
     'ing_last_6mths',
     'open_acc',
     'pub_rec',
     'revol_bal',
     'revol_util',
     'total_acc',
     'initial_list_status',
     'total_rec_late_fee',
     'last_pymnt_amnt',
     'last_fico_range_high',
     'last_fico_range_low',
     'collections_12_mths_ex_med',
     'policy_code',
     'application_type',
     'acc_now_dealing',
     'tot_coll_amt',
     'tot_cur_bal',
     'total_bal_il',
     'max_bal_bc',
     'issue_d_month',
     'earliest_cr_line_month',
     'earliest_cr_line_year',
     'last_pymnt_d_month',
     'last_credit_pull_d_month',
     'last_credit_pull_d_year']

    X = df[features]
    if st.button('Predict'):
        output = predict_grade(X)
        if output == 0:
            st.success('Grade is "A"')
        elif output == 1:
            st.success('Grade is "B"')
        elif output == 2:
            st.success('Grade is "C"')
        elif output == 3:
            st.success('Grade is "D"')
        elif output == 4:
            st.success('Grade is "E"')
        elif output == 5:
            st.success('Grade is "F"')
        else:
            st.success('Grade is "G"')
if __name__ == '__main__':
    main()
    