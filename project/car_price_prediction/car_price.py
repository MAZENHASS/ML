import os
import pickle
import pandas as pd
import streamlit as st
import pathlib
import gdown

BASE_DIR = pathlib.Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"

# Google Drive file ID for your model
MODEL_FILE_ID = "YOUR_FILE_ID_HERE"  # <-- replace with your own
MODEL_PATH = BASE_DIR / "car_price.pkl"

CSV_PATH = BASE_DIR / "cleaned.csv"
IMAGE_PATH = BASE_DIR / "car price.png"

# Download model if not found
if not MODEL_PATH.exists():
    st.info("Downloading model file from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", str(MODEL_PATH), quiet=False)

# Load model
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Load dataset
all = pd.read_csv(CSV_PATH)

# App setup
st.set_page_config(page_title="Car Price Prediction", layout="wide")

# Display image if exists
if IMAGE_PATH.exists():
    st.image(str(IMAGE_PATH))
else:
    st.warning(f"Image not found: {IMAGE_PATH}")

cg = all.copy()

# ===== Streamlit page UI =====
st.title('Car Price Prediction')
st.sidebar.header('Feature Selection')
st.sidebar.info('An easy app to predict')

def back(name, k1, k2, container):
    mapping = dict(zip(k1, k2))
    selected = container.selectbox(name, k1)
    return mapping[selected]

menu = []
col1, col2, col3 = st.columns(3)
d = 0

for i in all.select_dtypes(include='object').columns:
    if d % 2 == 0:
        menu.append(back(i, all[i].unique(), all[i + 'encoded'].unique(), col1))
    else:
        menu.append(back(i, all[i].unique(), all[i + 'encoded'].unique(), col2))
    d += 1

all = cg.copy()
all.drop('Price', inplace=True, axis=1)
del cg

col3.markdown('[Dataset from Kaggle](https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge/data)')

# ===== Numeric inputs =====
encoded_cols = [col + 'encoded' for col in all.select_dtypes(include='object').columns]
all[['Levy', 'Prod. year', 'Airbags']] = all[['Levy', 'Prod. year', 'Airbags']].astype(int)

for i in all.select_dtypes(exclude='object').columns:
    if i not in encoded_cols:
        container = col1 if d % 2 == 0 else col2
        if all[i].dtype == 'float':
            menu.append(container.number_input(
                i,
                min_value=float(all[i].min()),
                max_value=float(all[i].max()),
                value=float(all[i].min())
            ))
        else:
            menu.append(container.number_input(
                i,
                min_value=int(all[i].min()),
                max_value=int(all[i].max()),
                value=int(all[i].min())
            ))
        d += 1

# ===== Prediction =====
cols = [
    'Manufacturer', 'Model', 'Category', 'Fuel type', 'Gear box type',
    'Drive wheels', 'Color', 'Levy', 'Prod. year', 'Leather interior',
    'Engine volume', 'Mileage', 'Cylinders', 'Airbags'
]
bb = dict(zip(cols, menu))
wb = pd.DataFrame(bb, index=[0])

if st.sidebar.button('Predict'):
    price = model.predict(wb)
    st.sidebar.success("Price is " + str(price))
