import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Car Price Prediction", layout="wide")


with open('car_price.pkl', 'rb') as f:
    model = pickle.load(f)

all = pd.read_csv('cleaned.csv')
st.image('car price.png')

cg = all.copy()

st.title('Car Price Prediction')
st.sidebar.header('Feature Selection')
st.sidebar.info('An easy app to predict car prices.')

def back(name, original_vals, encoded_vals, container):
    mapping = dict(zip(original_vals, encoded_vals))
    selected = container.selectbox(name, original_vals)
    return mapping[selected]

menu = []
col1, col2 = st.columns(2)
d = 0

for i in all.select_dtypes(include='object').columns:
    if d % 2 == 0:
        menu.append(back(i, all[i].unique(), all[i + 'encoded'].unique(), col1))
    else:
        menu.append(back(i, all[i].unique(), all[i + 'encoded'].unique(), col2))
    d += 1

encoded_cols = [col + 'encoded' for col in all.select_dtypes(include='object').columns]
for i in all.select_dtypes(exclude='object').columns:
    if i not in encoded_cols and i != 'Price':
        if d % 2 == 0:
            menu.append(col1.number_input(i, float(all[i].min()), float(all[i].max()), float(all[i].min())))
        else:
            menu.append(col2.number_input(i, float(all[i].min()), float(all[i].max()), float(all[i].min())))
        d += 1


cols = ['Manufacturer', 'Model', 'Category', 'Fuel type', 'Gear box type',
        'Drive wheels', 'Color', 'Levy', 'Prod. year', 'Leather interior',
        'Engine volume', 'Mileage', 'Cylinders', 'Airbags']

bb = dict(zip(cols, menu))
wb = pd.DataFrame(bb, index=[0])

if st.sidebar.button('Predict'):
    price = model.predict(wb)[0]
    st.sidebar.success(f" Predicted Price: {price:,.2f}")

st.markdown("[Dataset from Kaggle](https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge/data)")
