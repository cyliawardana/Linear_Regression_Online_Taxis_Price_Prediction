import pandas as pd
import numpy as np
import streamlit as st
import joblib

with open('model_linreg.pkl', 'rb') as file_1:
  model_linreg = joblib.load(file_1)

with open('model_standardscaler.pkl', 'rb') as file_2:
  model_standardscaler = joblib.load(file_2)

with open('model_onehotencoder.pkl', 'rb') as file_3:
  model_onehotencoder = joblib.load(file_3)

with open('model_ordinalencoder.pkl','rb') as file_4:
  model_ordinalencoder = joblib.load(file_4)

st.title('Harga Prediksi Taksi Online')
jam = st.slider('Jam',0,23)
name = st.selectbox('Service Name',tuple(model_onehotencoder.categories_[0].tolist()))
#name = st.selectbox('Service Name',('Black','Black SUV',...))
distance = st.slider('Distance (miles)',0,20)
sm = st.selectbox('Surge Multiplier', (1.0,1.25,1.5,1.75,2.0,2.5,3.0))
st.write(type(sm))

df_inf = pd.DataFrame({
    'hour':[jam],
    'name':[name],
    'distance':[distance],
    'surge_multiplier':[sm]
})

'''
df_inf = pd.DataFrame([[jam,name,distance,sm]],columns=['hour','name','distance','surge_multiplier'])
'''
df_inf

df_inf_ohe = model_onehotencoder.transform(df_inf[['name']])
df_inf_ordenc = model_ordinalencoder.transform(df_inf[['surge_multiplier']])
df_inf_scaler = model_standardscaler.transform(df_inf[['hour','distance']])
df_inf_final = np.concatenate([df_inf_ohe,df_inf_ordenc,df_inf_scaler],axis=1)


if st.button('Predict'):
    df_inf_predict = model_linreg.predict(df_inf_final)
    st.subheader('Harga taksi: ${:.2f}'.format(df_inf_predict[0]))


