import pickle
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import io
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Insurance Visualization", page_icon="📊")
st.set_option('deprecation.showPyplotGlobalUse', False)


def load_data(nrows):
    data = pd.read_csv('insurance.csv', nrows=nrows)
    # One-hot encoding for nominal features
    catg_data = pd.get_dummies(data[['sex', 'smoker', 'region']])
    return data

st.write('''# Improve Predictive Performance Insurance''')
insur_data = load_data(1000)

# Example dataframe
df = pd.read_csv('insurance.csv')
st.subheader('Insurance Data')
st.write(insur_data)


st.subheader('Trend of Charge by Age')
df = pd.DataFrame(df[20:70], columns=['charges', 'age'])
st.line_chart(df , x ="age" , y= "charges")
st.info('Follow by age, charge has main relation with age in increasing when age of customers has been increase. That mean if you have older age, you will be need more than charge to paid.')

dfline2 = pd.DataFrame(insur_data[15:35], columns=['charges', 'bmi'])
st.subheader('Trend of Charge by BMI')
st.line_chart(dfline2, x ="bmi" , y= "charges")
st.info('BMI has trend similar with age as well as when compare with charge. If you have much BMI, you will get more charge of cost because who have higher BMI your health more risk than well.')


# dfplot.hist()
# st.subheader('Overview Infomation')
# st.pyplot()
# st.info('From this chart below, most customers have charge less than or equal 10,000$ and there have some customers have to paid cost more than most price of majority.And BMI of most customers less than 30')
# st.subheader('Trend of Charge by Age')
# st.line_chart(dfline)
# st.info('Follow by age, charge has main relation with age in increasing when age of customers has been increase. That mean if you have older age, you will be need more than charge to paid.')




st.sidebar.write('''# fill or select here''')
g_Sex = st.sidebar.radio('Gender :', ['male', 'female'])
g_Age = st.sidebar.number_input(
    'Age (Year) ', min_value=12, max_value=60, value=20)
g_BMI = st.sidebar.number_input('BMI (BMI)', value=20.00)
g_child = st.sidebar.slider(
    'Children Count (person)', min_value=0, max_value=5, value=1)
g_smoke = st.sidebar.radio('Do you smoke ', ['yes', 'no'])
g_region = st.sidebar.radio('Where do you live in American', [
                            'southwest', 'southeast', 'northwest', 'northeast'])



data = {
    'age': g_Age,
    'sex': g_Sex,
    'bmi': g_BMI,
    'children': g_child,
    'smoker': g_smoke,
    'region': g_region,
}



# Change the value of sex to be {'M','F','I'} as stored in the trained dataset

# if g_Sex == 'Male':
#     g_Sex = 'M'
# elif g_Sex == 'Female':
#     g_Sex = 'F'


# if g_smoke == 'yes' : g_smoke = 'Y'
# elif  g_smoke == 'no' : g_smoke = 'N'

# if g_region == 'southwest' : g_region = 'SW'
# elif  g_region == 'southeast' : g_region = 'SE'
# elif  g_region == 'northwest' : g_region = 'NW'
# elif  g_region == 'northeast' : g_region = 'NE'


df = pd.DataFrame(data, index=[0])
st.header('Application Improveore Prediction:')

st.subheader('User Input:')
st.write(df)


# Combines user input data with sample dataset
# The sample data contains unique values for each nominal features
# This will be used for the One-hot encoding
data_sample = pd.read_csv('insurance_sample.csv')
df = pd.concat([df, data_sample], axis=0)

# One-hot encoding for nominal features
cat_data = pd.get_dummies(df[['sex', 'smoker', 'region']])

# Combine all transformed features together
info = pd.concat([cat_data, df], axis=1)
info = info[:1]
# Select only the first row (the user input data)
# Drop un-used feature
info = info.drop(columns=['sex', 'smoker', 'region'])
# Show the X_new data frame on the screen
# st.subheader('Pre-Processed Input:')
# st.write(info)


# mms = MinMaxScaler()
# X = mms.fit_transform(info)
# normal = pd.DataFrame(X, columns=df.columns)
# normal.head()

# -- Reads the saved normalization model
load_nor = pickle.load(open('normalization.pkl', 'rb'))
# Apply the normalization model to new data
info = load_nor.transform(info)
# Show the X_new data frame on the screen
# st.subheader('Normalized Input:')
# st.write(info)

# -- Reads the saved classification model
load_rgt = pickle.load(open('regression.pkl', 'rb'))
# Apply model for prediction
predictionrgt = load_rgt.predict(info)
# Show the prediction result on the screen
st.subheader('Insurance Predicted Cost (Model Regression ):')  
st.write(predictionrgt)


