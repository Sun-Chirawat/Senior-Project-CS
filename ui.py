import pickle
import streamlit as st
import pandas as pd


st.set_page_config(page_title="Insurance Visualization", page_icon="📊")
st.set_option('deprecation.showPyplotGlobalUse', False)


def load_data(nrows):
    data = pd.read_csv('insurance.csv', nrows=nrows)
    # One-hot encoding for nominal features
    catg_data = pd.get_dummies(data[['sex', 'smoker', 'region']])
    return data

st.write('''# Improve Predictive Insurance Cost''')
insur_data = load_data(1000)

# Example dataframe
df = pd.read_csv('insurance.csv')
# st.subheader('Insurance Data')
# st.write(insur_data)


st.subheader('Trend of Charge by Age')
df = pd.DataFrame(df[20:70], columns=['charges', 'age'])
st.line_chart(df , x ="age" , y= "charges")
st.info('Follow by age, charge has main relation with age in increasing when age of customers has been increase. That mean if you have older age, you will be need more than charge to paid.')

dfline2 = pd.DataFrame(insur_data[15:35], columns=['charges', 'bmi'])
st.subheader('Trend of Charge by BMI')
st.line_chart(dfline2, x ="bmi" , y= "charges")
st.info('BMI has trend similar with age as well as when compare with charge. If you have much BMI, you will get more charge of cost because who have higher BMI your health more risk than well.')


st.sidebar.write('''# Please Insert Your Information: ''')
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


df = pd.DataFrame(data, index=[0])
st.header('Application Improve Prediction:')

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
st.subheader('Insurance Predicted Cost ($):')  
st.write(predictionrgt)


