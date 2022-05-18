from multiprocessing.sharedctypes import Value
from sklearn import preprocessing 
from pyparsing import And
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import numpy

st.write("""# Improve Predictive Performance Insurance""")

g_Sex = st.radio('Gender :', ['male', 'female'])
g_Age = st.slider('Age (Year) ', min_value=12, max_value=60, value=20)
g_BMI = st.slider('BMI (BMI)' , min_value=15.0, max_value=50.0, value=20.0)
g_child = st.slider('Children Count (person)', min_value=0, max_value=5, value=1)
g_smoke = st.radio('Do you smoke ', ['yes', 'no'])
g_region = st.radio('Where do you live in American', ['southwest', 'southeast', 'northwest', 'northeast'])



if g_Sex == 'male': g_charge = 10000+5000
elif g_Sex == 'female': g_charge = 10000


if g_Age <= 20 : g_charge +=5000
elif g_Age <= 30 : g_charge +=10000
elif g_Age <= 40 : g_charge +=15000
elif g_Age == 60 : g_charge +=20000

if g_BMI <= 18 : g_charge -= 5000
elif g_BMI <= 30 : g_charge +=10000
elif g_BMI <= 50 : g_charge += 15000

if g_smoke == 'yes' : g_charge += 10000
elif g_smoke == 'no': g_charge 

 


data = {
    'sex' : g_Sex,
    'age' : g_Age,
    'bmi' : g_BMI,
    'children' : g_child,
    'smoker' : g_smoke,
    'region' : g_region,
    'charge' : g_charge
    
}
# Create a data frame from the above dictionary
df = pd.DataFrame(data, index=[0])  
df = df.drop(columns=['charge'])
st.subheader('User Input : ')
st.write(df)

df2 = pd.DataFrame(data, index=[0]) 
prediction = df2.drop(columns=['sex','age','bmi','smoker','region','children'])

st.subheader('Charge : ')
st.write(prediction)
 
# Combines user input data with sample dataset
# The sample data contains unique values for each nominal features
# This will be used for the One-hot encoding
# data_sample = pd.read_csv('insurance_sample.csv')
# df = pd.concat([df, data_sample],axis=0)

# #One-hot encoding for nominal features
# cat_data = pd.get_dummies(df[['sex','smoker','region']])

# #Combine all transformed features together
# X_new = pd.concat([cat_data, df], axis=1)
# X_new = X_new[:1] # Select only the first row (the user input data)

# #Drop un-used feature
# X_new = X_new.drop(columns=['sex','smoker','region','charge'])

#Show the X_new data frame on the screen
# st.subheader('Pre-Processed Input:')
# st.write(X_new)

# data = X_new.values
# X_array = numpy.array(data)

# mms = preprocessing.MinMaxScaler()
# X_scale = mms.fit_transform(X_array)

# Nor_x = pd.DataFrame(X_scale, columns=X_new.columns)
# st.subheader('MMS')
# st.write(Nor_x)

# import pickle
# # -- Reads the saved classification model
# load_clu = pickle.load(open('Cluster_Regression.pkl','rb'))
# Apply model for prediction
# prediction = X_new
# #Show the prediction result on the screen
# st.subheader('Charge:')
# st.write(prediction)














