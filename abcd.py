import streamlit as st
import pandas as pd
import numpy as np

st.write("""
# Simple COVID-19 Prediction App
This app predicts whether you are infected with Covid-19 or not based on your symptoms!
""")

st.sidebar.header('User Input Parameters')

Breathing Problem = st.sidebar.selectbox(
    'Do you have Breathing Problem?',
     ['Yes','No'])
Fever = st.sidebar.selectbox(
    'Do you have Fever?',
     ['Yes','No'])
Dry Cough = st.sidebar.selectbox(
    'Do you have Dry Cough?',
     ['Yes','No'])
Sore throat = st.sidebar.selectbox(
    'Do you have Sore throat?',
     ['Yes','No'])
Running Nose = st.sidebar.selectbox(
    'Do you have Running Nose?',
     ['Yes','No'])
Asthma = st.sidebar.selectbox(
    'Do you have Asthma?',
     ['Yes','No'])
Chronic Lung Disease = st.sidebar.selectbox(
    'Do you have Chronic Lung Disease?',
     ['Yes','No'])
Headache = st.sidebar.selectbox(
    'Do you have Headache?',
     ['Yes','No'])
Heart Disease = st.sidebar.selectbox(
    'Do you have Heart Disease?',
     ['Yes','No'])
Diabetes = st.sidebar.selectbox(
    'Do you have Diabetes?',
     ['Yes','No'])
Hyper Tension = st.sidebar.selectbox(
    'Do you have Hyper Tension?',
     ['Yes','No'])
Fatigue = st.sidebar.selectbox(
    'Do you have Fatigue?',
     ['Yes','No'])
Gastrointestinal = st.sidebar.selectbox(
    'Do you have Gastrointestinal?',
     ['Yes','No'])
Abroad travel = st.sidebar.selectbox(
    'Do you have Abroad travel?',
     ['Yes','No'])
Contact with COVID Patient = st.sidebar.selectbox(
    'Do you have Contact with COVID Patient?',
     ['Yes','No'])
Attended Large Gathering = st.sidebar.selectbox(
    'Do you have Attended Large Gathering?',
     ['Yes','No'])
Visited Public Exposed Places = st.sidebar.selectbox(
    'Do you have Visited Public Exposed Places?',
     ['Yes','No'])
Family working in Public Exposed Places = st.sidebar.selectbox(
    'Do you have Family working in Public Exposed Places?',
     ['Yes','No'])
Wearing Masks = st.sidebar.selectbox(
    'Do you Wearing Masks?',
     ['Yes','No'])
Sanitization from Market = st.sidebar.selectbox(
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

