import streamlit as st
import pandas as pd
import numpy as np

st.write("""
# Simple COVID-19 Prediction App
This app predicts whether you are infected with Covid-19 or not based on your symptoms!
""")

st.sidebar.header('User Input Parameters')

BP = st.sidebar.selectbox(
    'Do you have Breathing Problem?',
     ['Yes','No'])
F = st.sidebar.selectbox(
    'Do you have Fever?',
     ['Yes','No'])
DC = st.sidebar.selectbox(
    'Do you have Dry Cough?',
     ['Yes','No'])
ST = st.sidebar.selectbox(
    'Do you have Sore throat?',
     ['Yes','No'])
RN = st.sidebar.selectbox(
    'Do you have Running Nose?',
     ['Yes','No'])
A = st.sidebar.selectbox(
    'Do you have Asthma?',
     ['Yes','No'])
CLD = st.sidebar.selectbox(
    'Do you have Chronic Lung Disease?',
     ['Yes','No'])
H = st.sidebar.selectbox(
    'Do you have Headache?',
     ['Yes','No'])
HD = st.sidebar.selectbox(
    'Do you have Heart Disease?',
     ['Yes','No'])
D = st.sidebar.selectbox(
    'Do you have Diabetes?',
     ['Yes','No'])
HT = st.sidebar.selectbox(
    'Do you have Hyper Tension?',
     ['Yes','No'])
F = st.sidebar.selectbox(
    'Do you have Fatigue?',
     ['Yes','No'])
G = st.sidebar.selectbox(
    'Do you have Gastrointestinal?',
     ['Yes','No'])
AT = st.sidebar.selectbox(
    'Do you have Abroad travel?',
     ['Yes','No'])
CWCP = st.sidebar.selectbox(
    'Do you have Contact with COVID Patient?',
     ['Yes','No'])
ALG = st.sidebar.selectbox(
    'Do you have Attended Large Gathering?',
     ['Yes','No'])
VPEP = st.sidebar.selectbox(
    'Do you have Visited Public Exposed Places?',
     ['Yes','No'])
FWIPEP = st.sidebar.selectbox(
    'Do you have Family working in Public Exposed Places?',
     ['Yes','No'])
WM = st.sidebar.selectbox(
    'Do you Wearing Masks?',
     ['Yes','No'])
SFM = st.sidebar.selectbox(
    'Do you Sanitization from Market?',
     ['Yes','No'])
    
    data = {'Breathing Problem': Breathing Problem,
            'Fever': Fever,
            'Dry Cough': Dry Cough,
            'Sore throat': Sore throat
            'Running Nose': Running Nose,
            'Asthma': Asthma,
            'Chronic Lung Disease': Chronic Lung Disease,
            'Headache': Headache
            'Heart Disease': Heart Disease,
            'Diabetes': Diabetes,
            'Hyper Tension': Hyper Tension,
            'Fatigue': Fatigue
            'Gastrointestinal': Gastrointestinal,
            'Abroad travel': Abroad travel,
            'Contact with COVID Patient': Contact with COVID Patient
            'Attended Large Gathering': Attended Large Gathering,
            'Visited Public Exposed Places': Visited Public Exposed Places,
            'Family working in Public Exposed Places': Family working in Public Exposed Places,
            'Wearing Masks': Wearing Masks
            'Sanitization from Market': Sanitization from Market}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

data = pd.read_csv('https://raw.githubusercontent.com/nuradrera/covid-19/main/Covid%20Dataset.csv')
X = data.drop('COVID-19', axis=1)
Y = data['COVID-19']

clf = RandomForestClassifier()
clf.fit(X, Y)
prediction = clf.predict(df)

st.subheader('Prediction')
# st.write(iris.target_names[prediction])
st.write(prediction)

