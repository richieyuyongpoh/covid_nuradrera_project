import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

st.write("""
# Simple COVID-19 Prediction App
This app predicts whether you are infected with Covid-19 or not based on your symptoms!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    A = st.sidebar.selectbox('Do you have Breathing Problem?',['Yes','No'])
    B = st.sidebar.selectbox('Do you have Fever?',['Yes','No'])
    C = st.sidebar.selectbox('Do you have Dry Cough?',['Yes','No'])
    D = st.sidebar.selectbox('Do you have Sore throat?',['Yes','No'])
    E = st.sidebar.selectbox('Do you have Running Nose?',['Yes','No'])
    F = st.sidebar.selectbox('Do you have Asthma?',['Yes','No'])
    G = st.sidebar.selectbox('Do you have Chronic Lung Disease?',['Yes','No'])
    H = st.sidebar.selectbox('Do you have Headache?',['Yes','No'])
    I = st.sidebar.selectbox('Do you have Heart Disease?',['Yes','No'])
    J = st.sidebar.selectbox('Do you have Diabetes?',['Yes','No'])
    K = st.sidebar.selectbox('Do you have Hyper Tension?',['Yes','No'])
    L = st.sidebar.selectbox('Do you have Fatigue?',['Yes','No'])
    M = st.sidebar.selectbox('Do you have Gastrointestinal?',['Yes','No'])
    N = st.sidebar.selectbox('Do you have Abroad travel?',['Yes','No'])
    O = st.sidebar.selectbox('Do you have Contact with COVID Patient?',['Yes','No'])
    P = st.sidebar.selectbox('Do you have Attended Large Gathering?',['Yes','No'])
    Q = st.sidebar.selectbox('Do you have Visited Public Exposed Places?',['Yes','No'])
    R = st.sidebar.selectbox('Do you have Family working in Public Exposed Places?',['Yes','No'])
    S = st.sidebar.selectbox('Do you Wearing Masks?',['Yes','No'])
    T = st.sidebar.selectbox('Do you Sanitization from Market?',['Yes','No'])
    
    data = {'Breathing Problem': A,
        'Fever': B,
        'Dry Cough': C,
        'Sore throat': D,
        'Running Nose': E,
        'Asthma': F,
        'Chronic Lung Disease': G,
        'Headache': H,
        'Heart Disease': I,
        'Diabetes': J,
        'Hyper Tension': K,
        'Fatigue': L,
        'Gastrointestinal': M,
        'Abroad travel': N,
        'Contact with COVID Patient': O,
        'Attended Large Gathering': P,
        'Visited Public Exposed Places': Q,
        'Family working in Public Exposed Places': R,
        'Wearing Masks': S,
        'Sanitization from Market': T}
    features = pd.DataFrame(data, index=[0])
    return features
data = pd.read_csv('https://raw.githubusercontent.com/nuradrera/covid-19/main/Covid%20Dataset.csv')

labelencoder1 = LabelEncoder() #kalau nak encoder lebih dari satu
labelencoder2 = LabelEncoder()
labelencoder3 = LabelEncoder()
labelencoder4 = LabelEncoder()
labelencoder5 = LabelEncoder()
labelencoder6 = LabelEncoder()
labelencoder7 = LabelEncoder()
labelencoder8 = LabelEncoder()
labelencoder9 = LabelEncoder()
labelencoder10 = LabelEncoder()
labelencoder11 = LabelEncoder()
labelencoder12 = LabelEncoder()
labelencoder13 = LabelEncoder()
labelencoder14 = LabelEncoder()
labelencoder15 = LabelEncoder()
labelencoder16 = LabelEncoder()
labelencoder17 = LabelEncoder()
labelencoder18 = LabelEncoder()
labelencoder19 = LabelEncoder()
labelencoder20 = LabelEncoder()

data['BreathingProblem'] = labelencoder1.fit_transform(data['BreathingProblem'])
data['Fever'] = labelencoder2.fit_transform(data['Fever'])
data['DryCough'] = labelencoder3.fit_transform(data['DryCough'])
data['SoreThroat'] = labelencoder4.fit_transform(data['SoreThroat'])
data['RunningNose'] = labelencoder5.fit_transform(data['RunningNose'])
data['Asthma'] = labelencoder6.fit_transform(data['Asthma'])
data['ChronicLungDisease'] = labelencoder7.fit_transform(data['ChronicLungDisease'])
data['Headache'] = labelencoder8.fit_transform(data['Headache'])
data['HeartDisease'] = labelencoder9.fit_transform(data['HeartDisease'])
data['Diabetes'] = labelencoder10.fit_transform(data['Diabetes'])
data['HyperTension'] = labelencoder11.fit_transform(data['HyperTension'])
data['Fatigue'] = labelencoder12.fit_transform(data['Fatigue'])
data['Gastrointestinal'] = labelencoder13.fit_transform(data['Gastrointestinal'])
data['AbroadTravel'] = labelencoder14.fit_transform(data['AbroadTravel'])
data['ContactWithCOVIDPatient'] = labelencoder15.fit_transform(data['ContactWithCOVIDPatient'])
data['AttendedLargeGathering'] = labelencoder16.fit_transform(data['AttendedLargeGathering'])

data['VisitedPublicExposedPlaces'] = labelencoder17.fit_transform(data['VisitedPublicExposedPlaces'])
data['FamilyWorkingInPublicExposedPlaces'] = labelencoder18.fit_transform(data['FamilyWorkingInPublicExposedPlaces'])
data['WearingMasks'] = labelencoder19.fit_transform(data['WearingMasks'])
data['SanitizationFromMarket'] = labelencoder20.fit_transform(data['SanitizationFromMarket'])

X = data.drop('COVID-19', axis=1)
Y = data['COVID-19']

df = user_input_features()

st.subheader('User Input parameters')
st.write(df.T)

# X = X.apply(LabelEncoder().fit_transform)

clf = RandomForestClassifier()
clf.fit(X, Y)
prediction = clf.predict(df)

st.subheader('Prediction')
# st.write(iris.target_names[prediction])
st.write(prediction)

