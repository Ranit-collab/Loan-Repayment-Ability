# Loan-Repayment-Ability-Using-Machine-Learning-DS-20-June
# TECHNOCOLABS  DATA ANALYSIS  INTERNSHIP PROJECT REPORT #  

_**TITLE:**_ 
  
Predicting Loan Repayment Ability with Grade Using Machine Learning and Deep Learning.  
  
_**AIM:**_  
  
The principle focus of our project is to perform data analysis and train a model using the most popular Machine Learning algorithm – Regularized Logistic Regression, Random Forest and Neural Networks in order to analyse the historical data that is present regarding the Loan Repayment.  
  
_**ABSTRACT:**_  
  
Evaluating and predicting the repayment ability of the loaners is important for the banks to minimize the risk of loan payment default. By this reason, there is a system created by the banks to process the loan request based on the loaners’ status, such as employment status, credit history, etc.. However, the current existing evaluation system might not be appropriate to evaluate some loaners repayment ability, such as students or people without credit histories. In order to properly assess the repayment ability of all groups of people, we trained various machine learning models on a Lending club dataset and evaluated the importance of all the features used. Then, based on the importance score of the features, we analyze and select the most identifiable features to predict the repayment ability of the loaner.
  
_**INTRODUCTION:**_  
  
Due to insufficient credit histories, many people are struggling to get loans from trustworthy sources, such as banks. These people are normally students or unemployed adults, who might not have enough knowledge to justify the credibility of the unidentified lenders. The untrustworthy lenders can take advantage of these borrowers by taking high interest rates or including hidden terms in the contract. Instead of evaluating the borrower based on their credit score, there are many other alternative ways to measure or predict their repayment abilities. For example, employment can be a big factor to affect the person’s repayment ability since an employed adult has more stable incomes and cash flow. Some other factors, such as real estates, marriage status and the city of residence, might also be useful in the study of the repayment ability. Therefore, in our project, we are planning to use machine learning algorithms to study the correlations between borrower status and their repayment ability.
We found the dataset from Lending club, to be used in this project. This open dataset contains 100K anonymous client’s with 152 unique features. By studying the correlation between these features and repayment ability of the clients, our algorithm can help lenders evaluate borrowers from more dimensions and can also help borrowers, especially those who do not have sufficient credit histories, to find credible loaner, leading to a win-win situation.  
  
_**OVERVIEW:**_  
  
 _Data Segmentation and Data Cleaning_  
  - Exploratory Data Analysis using python’s data visualisation libraries.  
  - Training the model based on the historical data available.  
  
**DATASET OVERVIEW:**  
  
For this project, we have taken Bank loan dataset from Lending Club. This dataset gives us the details about the loans that are either Fully Paid or Charged off by the customer. According to the dataset, there are 152 independent variables describing about a particular Loan Application. We analyzed Lending Club’s dataset of roughly 100k loans between 2007–18. We chose to only analyze loans that were paid off in full, charged off or defaulted in this case. Also the characteristics at time of application and loan characteristics at time of issuance.  

| File name      | Description | Description    |
| :---        |    :----:   |          ---: |
| accepted_2017_to_2018q4.csv      |    Information about loan accepted when they submit the application.    | 152   |
| rejected_2017_to_2018q4.csv   |    Information about loan rejected when they submit the application.     | 9      |  
  
**DATA SEGMENTATION AND DATA CLEANING:**
  
 -	In this project, we have prepared a processed dataset by and collected the clear-cut data available online.  
 -	Using pandas data frame, we have calculated the mean of every column.  
 -	We have dropped the columns that containing more than 70% of missing values.  
 -	By using the fillna we have filled all the cells with mean values for numeric data.  
 -	While doing so, we have not included the countries having a zero value in their cells.  
 -	We have manually replaced the zeros in a column with the mean of the column.  
 -	The original format of the file was XLSX. We have converted into CSV format and proceeded.  
  
## EXPLORATORY DATA ANALYSYIS:
  
We used Jupyter notebook and Python libraries (Matplotilb, Pandas, Seaborn) for data visualization.  
We first began to look at our data to better understand our demographics. We started by taking a look at the length of employment for our customers. We also explored the distribution of the loan amounts and see when did the loan amount issued increased significantly and were able to draw the following conclusion:  
 -	Most of the loans issued were in the range of 10,000 to 20,000 USD.  
 -	The year of 2015 was the year were most loans were issued.  
 -	Loans were issued in an incremental manner. (Possible due to a recovery in the U.S economy)  
 -	The loans applied by potential borrowers, the amount issued to the borrowers and the amount funded by investors are similarly distributed, meaning that it is most likely that qualified borrowers are going to get the loan they had applied for.  
Next, we took a look at what is the amount of bad loans Lending Club has declared so far keeping in mind that there were still loans that were at a risk of defaulting in the future. Also, the amount of bad loans could increment as the days pass by, since we still had a great amount of current loans. Average annual income was an important key metric for finding possible opportunities of investments in a specific region.  
  
__The conclusion that we drew from this were:__  
  
 -	Currently, bad loans consist 7.60% of total loans but remember that we still have current loans which have the risk of becoming bad loans. (So this percentage is subjected to possible changes.)  
 -	The NorthEast region seemed to be the most attractive in term of funding loans to borrowers.  
 -	The SouthWest and West regions have experienced a slight increase in the "median income" in the past years.  
 -	Average interest rates have declined since 2012 but this might explain the increase in the volume of loans.  
 -	Employment Length tends to be greater in the regions of the SouthWest and West  
 -	Clients located in the regions of NorthEast and MidWest have not experienced a drastic increase in debt-to-income(dti) as compared to the other regions.  
 -	Fully Paid loans tend to be smaller. This could be due to the age of the loans  
 -	Default has the highest count among other loan status.  
 -	In Grace Period and Late(16~30 days) have the highest loan amount and mean.
    
The next question we wanted to answer was "What kind of loans are being issued?". We decided to approach this by the grade that LendingClub assigns to the loan. The Grade is a value from A to G that is a culmination of LendingClub’s own analysis on the ability for the customer to repay the grade and the insights that we drew from this were:    
 -	Interest rate varied wildly, reaching nearly 30% for high-risk loans  
 -	Grade A has the lowest interest rate around 7%  
 -	Grade G has the highest interest rate above 25%
    
In the next part we analyzed loans issued by region in order to see region patters that will allow us to understand which region gives Lending Club.
  
__Summary:__  
  
 -	South-East , West and North-East regions had the highest amount lof loans issued.  
 -	West and South-West had a rapid increase in debt-to-income starting in 2012.  
 -	West and South-West had a rapid decrease in interest rates (This might explain the increase in debt to income) 
   
Deeper Look into Bad Loans: 
  
We looked at the number of loans that were classified as bad loans for each region by its loan status. We also had a closer look at the operative side of business by state. This gave us a clearer idea in which state we have a higher operating activity We focused on three key metrics: Loans issued by state (Total Sum), Average interest rates charged to customers and average annual income of all customers by state. The purpose of this analysis was to see states that give high returns at a descent risk.  
  
__And we concluded as follows:__  
  
 -	The regions of the West and South-East had a higher percentage in most of the "bad" loan statuses.  
 -	The North-East region had a higher percentage in Grace Period and Does not meet Credit Policy loan status. However, both of these are not considered as bad as default for instance.  
 -	Based on this small and brief summary we can conclude that the West and South-East regions have the most undesirable loan status, but just by a slightly higher percentage compared to the North-East region.  
 -	California, Texas, New York and Florida were the states in which the highest amount of loans were issued.  
 -	Interesting enough, all four states had an approximate interest rate of 13% which is at the same level of the average interest rate for all states (13.24%)  
 -	California, Texas and New York were all above the average annual income (with the exclusion of Florida), this gave possible indication why most loans were issued in these states. 
   
## DATA MODELLING :   
  
__Team-B’s :__  After further analysis and cleaning of data and filling missing values we came down to 2011813 rows × 91 columns of our data and carefully observing the relevance, we out of that selected 33 features  and one target value as listed below:  

```output
 1.	'funded_amnt_inv',  
 2.	'int_rate',  
 3.	'emp_length',  
 4.	'annual_inc',  
 5.	'pymnt_plan',  
 6.	'dti',  
 7.	'deling_2yrs',  
 8.	'fico_range_low',  
 9.	'ing_last_6mths',  
 10.	'open_acc',  
 11.	'pub_rec',  
 12.	'revol_bal',  
 13.	'revol_util',  
 14.	'total_acc',  
 15.	'initial_list_status',  
 16.	'total_rec_late_fee',  
 17.	'last_pymnt_amnt',  
 18.	'last_fico_range_high',  
 19.	'last_fico_range_low',  
 20.	'collections_12_mths_ex_med',  
 21.	'policy_code',  
 22.	'application_type',  
 23.	'acc_now_dealing',  
 24.	'tot_coll_amt',  
 25.	'tot_cur_bal',  
 26.	'total_bal_il',  
 27.	'max_bal_bc',  
 28. Datetime columns: 'issue_d_month', 'earliest_cr_line_month','earliest_cr_line_year','last_pymnt_d_month', 'last_credit_pull_d_month','last_credit_pull_d_year'__     
``` 
**Target value: 'grade'**

Next, after scaling our data using “standard scaler” we split our data into training data, validation data and test data with a ratio of 9:1 (90% training data and 10% validation and test data). Going further we first employed Regularized Logistic Regression and created a  Baseline model. After fitting our training data we checked for its accuracy and it came out to be  86.74% , so in order to increase its accuracy we went for Hyper-parameterisation using Grid search CV module  and got our best parameters but there was no drastic change in accuracy. We also calculated precision, recall and F1 score using baseline model on test data and finally the confusion matrix was made and plotted for various classes.  
So, we went on to use a different ML algo this time. We used  Random Forest  and after fitting our data and tuning it using Hyper-parameterisation we got an excellent accuracy of 92.03%. We again calculated the same things as mentioned above. Next we calculated AUC (Area Under the Curve) for both regularized logistic regression model and random forest model and it came out to be 98.68% and 99.7% resp. We also plotted ROC curve for both our models.   
  
## DEPLOYMENT: 
  
__Team-B’s :__ We finally chose to work with Neural Networks for our model which are computing systems inspired by the biological neural networks that constitute animal brains. An NN is based on a collection of connected units or nodes called artificial neurons, which loosely model the neurons in a biological brain. Each connection, like the synapses in a biological brain, can transmit a signal to other neurons. An artificial neuron that receives a signal then processes it and can signal neurons connected to it.   
For our model, Sequential is a model. Dense is a layer accepting no. of hidden units in each layer and some activation. Our model has one hidden unit only. The hidden unit has 64 units and an activation of relu. The output unit has 7 units has we have predict 7 classes A,B,C....,G and the loss is categorical cross-entropy. We are using categorical cross-entropy as output has 7 classes. Also the activation in output layer is softmax. What a softmax function does is it's sums up the probability of all classes to one.  
![__Softmax__:](data:image/png;base64,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)  
![__NN model with one hidden layer:__](data:image/jpeg;base64,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)  
   
Finally, we made a .h5 file of our model and used cloud services of Heroku to finally create our web application. 
  
Deployed Online url : [https://loangradeprediction.herokuapp.com/](https://loangradeprediction.herokuapp.com/)  
  
# Logistic Regession to Predict Loan Grade
**Logistic Regression** is a Machine Learning classification algorithnm that is used to predict the probability of categorical dependent variable.In logistic regression,the dependent variable is a **binary** or **Multinomial**, which involves having more than one category.

## Logistic Regression Assumptions:
- The independent variables should be independent of each other. That is, the model should have little or no multicollinearity.
- The independent variables are linearly related to the log odds.
- Logistic regression requires quite large sample sizes.

> **🧮 Mathematics behind logistic regression**
>
> Remember how linear regression often used ordinary least squares to arrive at a value? Logistic regression relies on the concept of 'maximum likelihood' using [sigmoid functions](https://wikipedia.org/wiki/Sigmoid_function). A 'Sigmoid Function' on a plot looks like an 'S' shape. It takes a value and maps it to somewhere between 0 and 1. Its curve is also called a 'logistic curve'. Its formula looks like thus:
>
> ![logi3](https://user-images.githubusercontent.com/78952426/126063661-af40a4fb-cd81-42b5-8eec-4b40ac681f01.png)
>
> where the sigmoid's midpoint finds itself at x's 0 point, L is the curve's maximum value, and k is the curve's steepness. If the outcome of the function is more than 0.5, the label in question will be given the class '1' of the binary choice. If not, it will be classified as '0'.
 
## Problem Statement
The data set contains Loan grades:7 classes where each class refers to digit(1:A,2:B,3:C,4:D,5:E,6:F,G:7).Objective of our model is to predict the correct grade ,based on given data and deploy that model to predict Loan Grade on Heroku using Flask.

## Build Multiclass Logistic Regression Model
Building a model to predict these multiclass is straightforward in Scikit-learn.
 - **Create Test and Train Dataset**

   select variables for classifiaction model and split dataset,so that we can use one set of data for training the model and one set for testing the model,split the training        and test sets calling `train_test_split()`:
   ```python
    from sklearn.model_selection import train_test_split
    
    features = [['funded_amnt_inv', 'int_rate', 'grade', 'emp_length','annual_inc', 'pymnt_plan', 'dti', 'deling_2yrs', 'fico_range_low','ing_last_6mths', 
    'open_acc', 'pub_rec', 'revol_bal', 'revol_util','total_acc', 'initial_list_status', 'total_rec_late_fee','last_pymnt_amnt', 'last_fico_range_high',
    'last_fico_range_low','collections_12_mths_ex_med', 'policy_code', 'application_type','acc_now_dealing', 'tot_coll_amt', 'tot_cur_bal', 'total_bal_il',
     'max_bal_bc', 'issue_d_month', 'earliest_cr_line_month','earliest_cr_line_year', 'last_pymnt_d_year','last_credit_pull_d_month', 'last_credit_pull_d_year']]
    
   X=features.drop(['grade'],axis=1)
   y=features['grade']
    # split data into training and testing data, for both features and target
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size= 0.2, random_state=1)
    ```
  - **Normalization of independent features**
  
    we have normalize our data using using `StandardScaler()`:
    ```python
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    train_X = ss.fit_transform(train_X)
    test_X = ss.fit_transform(test_X)
    ```
    Now train the model, by calling `fit()` with training data:
    ```python
    from sklearn.metrics import accuracy_score,confusion_matrix, classification_report 
    from sklearn.linear_model import LogisticRegression
     accepted_model = LogisticRegression(penalty='none')
     accepted_model.fit(train_X, train_y)
     val_predictions = accepted_model.predict(test_X)
    
    ```
**Model Score**

Check the model score for training and test data:
 ```python
 print('Accuray: {}'.format(accepted_model.score(train_X,train_y)))
print("Accuracy: {}".format(accepted_model.score(test_X, test_y)))
```
```output
Accuray: 0.8697499145670882
Accuracy: 0.8702987103684982
```
**Confusion Matrix**
- Confusion Matrix helps us to visualize the performance of model.
- The diagonal elements represent the number of points for which the predicted label is equal to true label.
- Off-diagonal elements are those that are mislabeled by the classifier.
- The heigher the diagonal values of confusion matrix the better indicating many correct.
SO for our model confusion matrix is-
```python
confusion = confusion_matrix(test_y, val_predictions)
print(" \n Confusion Matrix is : \n", confusion)
```
```output
Confusion Matrix is : 
 [[ 75116   4178      0      0      0      0      0]
 [  5511 105556   7105      0      0      0      0]
 [    40   6860 103954   4483      0      0      0]
 [    27      0   6297  46486   3950      0      0]
 [     8      0      1   7558  14610   1211      0]
 [     6      0      0     11   3020   3746    503]
 [     1      0      0      0     87   1330    708]]
```
**Classification Report**

Classification report is used to measure the quality of prediction from classification algorithm
- Precision:Indiactes how many classes are correctly classified.
- Recall:Indicates what proportions of actual positives was identified correctly.
- F-score:It is the harmonic mean between precision & recall.
- Support:It is the number of occurences of the given class in our dataset.
```python
report = classification_report(test_y, val_predictions)
print(" \n Classification Report is : \n", report)
```
```output
Classification Report is : 
               precision    recall  f1-score   support

           1       0.93      0.95      0.94     79294
           2       0.91      0.89      0.90    118172
           3       0.88      0.90      0.89    115337
           4       0.79      0.82      0.81     56760
           5       0.66      0.61      0.63     23388
           6       0.55      0.46      0.50      7286
           7       0.57      0.33      0.42      2126

    accuracy                           0.87    402363
   macro avg       0.75      0.71      0.73    402363
weighted avg       0.86      0.87      0.87    402363
```

## Visualize the ROC curve of this model

This is not a bad model; its accuracy is in the 87% range so ideally you could use it to predict the loan grade given a set of variables.

Let's do one more visualization to see the so-called 'ROC' score:
```python
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# Get class probability scores
grade_prob = accepted_model.predict_proba(test_X)

# Get ROC metrics for each class
fpr = {}
tpr = {}
thresh ={}
for i in range(len(grade)):
  fpr[i], tpr[i], thresh[i] = roc_curve(test_y, grade_prob[:,i], pos_label=i) 
# Plot the ROC chart
plt.figure(figsize=(10,10))
plt.plot(fpr[0], tpr[0],color='orange', label=str(grade[0]) + ' vs Rest')
plt.plot(fpr[1], tpr[1],color='green', label=str(grade[1]) + ' vs Rest')
plt.plot(fpr[2], tpr[2],color='blue', label=str(grade[2]) + ' vs Rest')
plt.plot(fpr[3], tpr[3],color='red', label=str(grade[3]) + ' vs Rest')
plt.plot(fpr[4], tpr[4],color='yellow', label=str(grade[4]) + ' vs Rest')
plt.plot(fpr[5], tpr[5],color='pink', label=str(grade[5]) + ' vs Rest')
plt.plot(fpr[6], tpr[6],color='cyan', label=str(grade[6]) + ' vs Rest')
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show()
```
ROC curves are often used to get a view of the output of a classifier in terms of its true vs. false positives. "ROC curves typically feature true positive rate on the Y axis, and false positive rate on the X axis."
![roc curve](https://user-images.githubusercontent.com/78952426/126062638-7c34bbb3-9bf7-44e5-9e3c-5fd0bcd411f9.png)
to compute the actual 'Area Under the Curve' (AUC):
```python
roc_auc = roc_auc_score(test_y, grade_prob, multi_class=('ovr'))
print(" \n AUC using ROC is : ", roc_auc)
```
```output
AUC using ROC is :  0.9865396293036989
```
The result is  0.9865396293036989. Given that the AUC ranges from 0 to 1, you want a big score, since a model that is 100% correct in its predictions will have an AUC of 1; in this case, the model is pretty good.

## **DEPLOYMENT AND WEB APPLICATION:**
  - What is mean by model deployment? Deploying a machine learning model, known as model deployment, simply means to integrate a machine learning model and integrate it into an existing production environment, where it can take in an input and return an output. 
  - The Lending Club’s Loan Grade Prediction Machine Learning Model is trained and tested using Logistic Regression with 87.02% accuracy. And using flask framework, HTML, CSS and python all the necessary files have created along with Procfile and requirement.txt. 
  - Using Heroku, Platform as a service we have successfully deployed our model in the online web server and provided ease to end users.
  - Deployed Online url:[http://loangradeprediction-api.herokuapp.com/](http://loangradeprediction-api.herokuapp.com/)
 


