import pickle
import streamlit as st
import pandas as pd

# Load model and data
st.set_page_config(page_title="Car Price Prediction", layout="wide")

model = pickle.load(open(r"C:\Users\User\Desktop\proj\car_price.pkl", 'rb'))
all = pd.read_csv(r'C:\Users\User\PycharmProjects\PythonProject4\.venv\cleaned.csv')
cg=all.copy()
# Streamlit page
st.title('Car Price Prediction')
st.sidebar.header('Feature Selection')
st.sidebar.info('An easy app to predict')
st.image(r"C:\Users\User\PycharmProjects\PythonProject4\.venv\66c26746-c655-4110-81eb-ce6854a9c5a4.png")

def back(name, k1, k2, container):
    mapping = dict(zip(k1, k2))
    selected = container.selectbox(name, k1)
    return mapping[selected]

menu = []
col1, col2,col3 = st.columns(3)
d = 0

for i in all.select_dtypes(include='object').columns:
    if d % 2 == 0:
        menu.append( back(i, all[i].unique(), all[i + 'encoded'].unique(), col1))
    else:
        menu.append(back(i, all[i].unique(), all[i + 'encoded'].unique(), col2))
    d += 1

    all=all[all['Manufacturerencoded'] ==menu[0]]
all=cg.copy()

all.drop('Price',inplace=True,axis=1)
del cg

col3.markdown('(Dataset from kaggle )[https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge/data]')
encoded_cols = [col + 'encoded' for col in all.select_dtypes(include='object').columns]
all[['Levy', 'Prod. year', 'Airbags']] = all[['Levy', 'Prod. year', 'Airbags']].astype(int)
intnum=['Levy', 'Prod. year', 'Airbags']
for i in all.select_dtypes(exclude='object').columns:
       if i not in encoded_cols  :
              if d % 2 == 0:
                     if all[i].dtype=='float':
                            menu.append( col1.number_input(i, min_value=float(all[i].min()), max_value=float(all[i].max()),
                                                        value=float(all[i].min())))
                     else:
                            menu.append(col1.number_input(i, min_value=int(all[i].min()), max_value=int(all[i].max()),
                                                        value=int(all[i].min())))


              else:

                     if all[i].dtype == 'float':
                            menu.append( col2.number_input(i, min_value=float(all[i].min()), max_value=float(all[i].max()),
                                                        value=float(all[i].min())))
                     else:
                            menu.append(col2.number_input(i, min_value=int(all[i].min()), max_value=int(all[i].max()),
                                                        value=int(all[i].min())))
              d+=1

cols=['Manufacturer', 'Model', 'Category', 'Fuel type', 'Gear box type',
       'Drive wheels', 'Color', 'Levy', 'Prod. year', 'Leather interior',
       'Engine volume', 'Mileage', 'Cylinders', 'Airbags']
bb=dict(zip(cols,menu))
wb = pd.DataFrame(bb, index=[0])
click= st.sidebar.button('predict')
if click :
       price = model.predict(wb)
       st.sidebar.success("Price is " + str(price))
