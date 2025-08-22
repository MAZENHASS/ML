import pickle
import streamlit as st
import pandas as pd

# Load model and data
model = pickle.load(open('car_price.pkl', 'rb'))

all = pd.read_csv('cleaned.csv')
cg=all.copy()
# Streamlit page
st.set_page_config(page_title="Car Price Prediction", layout="wide")


st.title('Car Price Prediction')
st.sidebar.header('Feature Selection')
st.sidebar.info('An easy app to predict')
st.image('car price.png')

def back(name, k1, k2, container):
    mapping = dict(zip(k1, k2))
    selected = container.selectbox(name, k1)
    return mapping[selected]

menu = {}
col1, col2 ,col3= st.columns(3)
d = 0

for i in all.select_dtypes(include='object').columns:
    if d % 2 == 0:
        menu[i + 'encoded'] = back(i, all[i].unique(), all[i + 'encoded'].unique(), col1)
    else:
        menu[i + 'encoded'] = back(i, all[i].unique(), all[i + 'encoded'].unique(), col2)
    d += 1

    all=all[all['Manufacturerencoded'] ==menu[ 'Manufacturerencoded']]
all=cg.copy()

all.drop('Price',inplace=True,axis=1)
del cg


encoded_cols = [col + 'encoded' for col in all.select_dtypes(include='object').columns]
all[['Levy', 'Prod. year', 'Airbags']] = all[['Levy', 'Prod. year', 'Airbags']].astype(int)
intnum=['Levy', 'Prod. year', 'Airbags']
for i in all.select_dtypes(exclude='object').columns:
       if i not in encoded_cols  :
              if d % 2 == 0:
                     if all[i].dtype=='float':
                            menu[i] = col1.number_input(i, min_value=float(all[i].min()), max_value=float(all[i].max()),
                                                        value=float(all[i].min()))
                     else:
                            menu[i] = col1.number_input(i, min_value=int(all[i].min()), max_value=int(all[i].max()),
                                                        value=int(all[i].min()))


              else:

                     if all[i].dtype == 'float':
                            menu[i] = col2.number_input(i, min_value=float(all[i].min()), max_value=float(all[i].max()),
                                                        value=float(all[i].min()))
                     else:
                            menu[i] = col2.number_input(i, min_value=int(all[i].min()), max_value=int(all[i].max()),
                                                        value=int(all[i].min()))
              d+=1

col3.markdown('(Dataset from kaggle )[https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge/data]')
wb=pd.DataFrame(menu,index=[0])

click=st.sidebar.button('Predict ')
if click:

       wb_predict=model.predict(wb.values)
       st.sidebar.success(wb_predict)

