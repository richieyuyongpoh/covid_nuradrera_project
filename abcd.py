import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

st.write("""
# Simple COVID-19 Prediction App
This app predicts whether you are infected with Covid-19 or not based on your symptoms!
""")

st.sidebar.write('**User Input Parameters**')
st.sidebar.write('Note: 1 for YES, 0 for NO')

def user_input_features():
    A = st.sidebar.selectbox('Do you have Breathing Problem?',['1','0'])
    B = st.sidebar.selectbox('Do you have Fever?',['1','0'])
    C = st.sidebar.selectbox('Do you have Dry Cough?',['1','0'])
    D = st.sidebar.selectbox('Do you have Sore throat?',['1','0'])
    E = st.sidebar.selectbox('Do you have Running Nose?',['1','0'])
    F = st.sidebar.selectbox('Do you have Asthma?',['1','0'])
    G = st.sidebar.selectbox('Do you have Chronic Lung Disease?',['1','0'])
    H = st.sidebar.selectbox('Do you have Headache?',['1','0'])
    I = st.sidebar.selectbox('Do you have Heart Disease?',['1','0'])
    J = st.sidebar.selectbox('Do you have Diabetes?',['1','0'])
    K = st.sidebar.selectbox('Do you have Hyper Tension?',['1','0'])
    L = st.sidebar.selectbox('Do you have Fatigue?',['1','0'])
    M = st.sidebar.selectbox('Do you have Gastrointestinal?',['1','0'])
    N = st.sidebar.selectbox('Do you have Abroad travel?',['1','0'])
    O = st.sidebar.selectbox('Do you have Contact with COVID Patient?',['1','0'])
    P = st.sidebar.selectbox('Do you have Attended Large Gathering?',['1','0'])
    Q = st.sidebar.selectbox('Do you have Visited Public Exposed Places?',['1','0'])
    R = st.sidebar.selectbox('Do you have Family working in Public Exposed Places?',['1','0'])
    S = st.sidebar.selectbox('Do you Wearing Masks?',['1','0'])
    T = st.sidebar.selectbox('Do you Sanitization from Market?',['1','0'])
    
    data = {
        'BreathingProblem': A,
        'Fever': B,
        'DryCough': C,
        'SoreThroat': D,
        'RunningNose': E,
        'Asthma': F,
        'ChronicLungDisease': G,
        'Headache': H,
        'HeartDisease': I,
        'Diabetes': J,
        'HyperTension': K,
        'Fatigue': L,
        'Gastrointestinal': M,
        'AbroadTravel': N,
        'ContactWithCOVIDPatient': O,
        'AttendedLargeGathering': P,
        'VisitedPublicExposedPlaces': Q,
        'FamilyWorkingInPublicExposedPlaces': R,
        'WearingMasks': S,
        'SanitizationFromMarket': T}
    features = pd.DataFrame(data, index=[0])
    return features
data = pd.read_csv('https://github.com/nuradrera/covid-19/blob/main/Covid%20Dataset.csv')

labelencoder1 = LabelEncoder()
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

X = data.drop('COVID19', axis=1)
Y = data['COVID19']

df = user_input_features()

st.subheader('User Input parameters')
st.write(df.T)

# X = X.apply(LabelEncoder().fit_transform)

clf = RandomForestClassifier()
clf.fit(X, Y)
prediction = clf.predict(df)

st.subheader('Your prediction to have Covid-19 is:')
# st.write(iris.target_names[prediction])
st.write(prediction)

