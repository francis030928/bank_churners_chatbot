
import streamlit as st 
# from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score as accuracy
import warnings
warnings.filterwarnings('ignore')


bank_ch = pd.read_csv('BankChurners.csv')
bank_ch.head()

bank_ch.drop(['CLIENTNUM','Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
          'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'], axis = 1, inplace = True)
bank_ch.head()


num = bank_ch.select_dtypes(exclude  = 'object') 
cat = bank_ch.select_dtypes(include = 'object')
num.head()
cat.head()


df = bank_ch.copy()

lb = LabelEncoder()

for i in cat:
  df[i] = lb.fit_transform(df[i])


x = df.drop(['Attrition_Flag'], axis = 1)
y = df.Attrition_Flag


scaler = StandardScaler() 
df_scaled = pd.DataFrame()
for i in x.columns: 
  df_scaled[[i]] = scaler.fit_transform(x[[i]])

df_scaled.head(3)



# ANOVA F-value between label/feature
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
best_feature1 = SelectKBest(score_func = f_classif, k = 'all')
fitting1 = best_feature1.fit(x,y)
scores1 = pd.DataFrame(fitting1.scores_)
columns1 = pd.DataFrame(x.columns)
feat_score1 = pd.concat([columns1, scores1], axis = 1)
feat_score1.columns = ['Feature', 'F_classif_score'] 
k1 = feat_score1.nlargest(6, 'F_classif_score')

k1.sort_values(by = 'F_classif_score', ascending = False)

sel_feat = ['Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Total_Revolving_Bal', 'Contacts_Count_12_mon', 'Avg_Utilization_Ratio', 'Total_Trans_Amt']

sel_feat = x[sel_feat]
sel_feat.head(3)


x_train, x_test, y_train, y_test = train_test_split( sel_feat, y, test_size = 0.15, random_state = 4, stratify = y)
print(f"X Train rows and column: {x_train.shape}")
print(f"Y Train rows and column: {y_train.shape}")
print(f"X Test rows and column: {x_test.shape}")
print(f"Y Test rows and column: {y_test.shape}")

from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()

# Model Creation
logistic.fit(x_train, y_train)

# Predict the test data for checking accuracy
prediction = logistic.predict(x_test)


import sklearn
from sklearn.ensemble import RandomForestClassifier
rfm = RandomForestClassifier()
rfm_model = rfm.fit(x_train, y_train)
cross_validate = rfm_model.predict(x_train)
sklearn.metrics.accuracy_score(y_train, cross_validate)

# Checking for the Accuracy of the Model Using Classification Report
from sklearn.metrics import classification_report
report = classification_report(y_test, prediction)
print(f'the score is {report} Good Job')
print(df.head())



import joblib
joblib.dump(rfm, 'Logistic_Model.pkl')


# FROM HERE WE BEGIN THE IMPLEMENTATION FOR STREAMLIT.

from PIL import Image
image = Image.open(r'churn_analysis.jpg')

st.header('BANK CHURNERS PREDICTION')
user_name = st.text_input('Register User')

if(st.button('SUBMIT')):
    st.text(f"You are welcome {user_name}. Enjoy your usage")

st.write(sel_feat)


image = Image.open(r'image_entry.jpg')
st.sidebar.image(image)

st.sidebar.subheader(f"Hey {user_name}")
independent_features = st.sidebar.radio('How do you want your feature input?\n \n \n', ('slider', 'direct input'))


if independent_features == 'slider':
   Total_Trans_Ct = st.sidebar.slider('Total Trans Ct', 10.0, 139.0, (5.0))

   Total_Ct_Chng_Q4_Q1 = st.sidebar.slider('Total Ct Chng Q4 Q1', 0.0, 3.7, (0.2))
 
   Total_Revolving_Bal = st.sidebar.slider('Total Revolving Bal', 0.0, 2517.0, (100.0))

   Contacts_Count_12_mon = st.sidebar.slider('Contacts Count 12 mon', 0.0, 6.0, (1.0))   

   Avg_Utilization_Ratio = st.sidebar.slider('Avg Utilization Ratio', 0.0, 0.99, (0.1))

   Total_Trans_Amt = st.sidebar.slider('Total Trans Amt', 510.0, 18484.0, (1.0))


else:
    Total_Trans_Ct = st.sidebar.number_input('Total Trans Ct')
    Total_Ct_Chng_Q4_Q1 = st.sidebar.number_input('Total Ct Chng Q4 Q1')
    Total_Revolving_Bal = st.sidebar.number_input('Total Revolving Bal')
    Contacts_Count_12_mon = st.sidebar.number_input('Contacts Count 12 mon')
    Avg_Utilization_Ratio = st.sidebar.number_input('Avg Utilization Ratio')
    Total_Trans_Amt = st.sidebar.number_input('Total Trans Amt')
     

st.write('selected inputs: ', [Total_Trans_Ct, Total_Ct_Chng_Q4_Q1, Total_Revolving_Bal, Contacts_Count_12_mon, Avg_Utilization_Ratio, Total_Trans_Amt])

input_values = [[Total_Trans_Ct, Total_Ct_Chng_Q4_Q1, Total_Revolving_Bal, Contacts_Count_12_mon, Avg_Utilization_Ratio, Total_Trans_Amt]]


# Modelling
# import the model
model = joblib.load(open('Logistic_Model.pkl', 'rb'))
pred = model.predict(input_values)


if pred == 0:
    st.success('This is an Attrited Customer')
else:
    st.success('This is an Existing Customer')
