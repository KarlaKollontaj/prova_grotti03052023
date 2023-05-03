import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import  mlem
import seaborn as sns

def main ():
    st.title ("prova Grotti 03/05/2023")

    df_house = pd.read_csv('https://frenzy86.s3.eu-west-2.amazonaws.com/python/data/formart_house.csv')
    df_house.rename(columns={"medv":"price"}, inplace=True)
    df_house.drop(df_house.tail(1).index, inplace=True)
    s = df_house.select_dtypes(include='object').columns
    df_house[s] = df_house[s].astype("float")

    fig, ax = plt.subplots()
    sns.heatmap(data=df_house.corr(), annot=True)
    st.write(fig)
    
    crim = st.number_input('Insert crim numero', 0, 10000, 5000)
    zn = st.number_input('Insert zn', 0, 10000, 5000)
    indus = st.number_input('Insert indus', 0, 10000, 5000)
    chas = st.number_input('Insert chas', 0, 10000, 5000)
    nox = st.number_input('Insert nox', 0, 10000, 5000)
    rm = st.number_input('Insert rm', 0, 10000, 5000)
    age = st.number_input('Insert age', 0, 10000, 5000)
    dis = st.number_input('Insert dis', 0, 10000, 5000)
    rad = st.number_input('Insert rad', 0, 10000, 5000)
    tax = st.number_input('Insert tax', 0, 10000, 5000)
    ptRatio = st.number_input('Insert ptRatio', 0, 10000, 5000)
    b = st.number_input('Insert b', 0, 10000, 5000)
    Istat = st.number_input('Insert Istat', 0, 10000, 5000)

    
    new_model = mlem.api.load('model_.mlem')
    pred = new_model.predict([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptRatio, b, Istat]])
    st.write('price: ', pred[0])

if __name__=="__main__":
    main() 