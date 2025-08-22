import pickle
import streamlit as st
import pandas as pd


all = pd.read_csv('cleaned.csv')
cg=all.copy()
# Streamlit page
st.set_page_config(page_title="Car Price Prediction", layout="wide")


st.title('Car Price Prediction')
st.sidebar.header('Feature Selection')
st.sidebar.info('An easy app to predict')
st.image('car price.png')

