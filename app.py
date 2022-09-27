import streamlit as st
import numpy as np
import pandas as pd
#from sklearn.linear_model import LinearRegression
import pickle
import matplotlib.pyplot as plt
import mpld3
import streamlit.components.v1 as components

st.set_page_config(
        page_title="Stage d'application",
        page_icon="2560px-Yazaki_company_logo.svg.png",
        layout="wide",
    )
from PIL import Image
image = Image.open('2560px-Yazaki_company_logo.svg.png')

st.image(image, caption='Yazaki meknès')    
st.markdown(
        "Dans le cadre d'un stage d'application au sein de l'entreprise Yazaki mèknes et sous l'encadrement du monsieur Maaroufi wassim et mon professeur Hossni moahmed je vous présente cette application qui vise à prédire le stock du mois prochain dans le département maintenance   "
    )

st.subheader('Prédiction de stock pour le mois prochain')
st.subheader('veuillez entrer le stock des 7 mois précédents ')

regr = pickle.load(open('forest_model.pkl', 'rb'))


Mois_1 = st.number_input("1er mois", min_value=0 )
Mois_2 = st.number_input("2ème mois", min_value=0 )
Mois_3 = st.number_input("3ème mois", min_value=0 )
Mois_4 = st.number_input("4 ème mois", min_value=0 )
Mois_5 = st.number_input("5ème mois", min_value=0 )
Mois_6 = st.number_input("6ème mois", min_value=0 )
Mois_7 = st.number_input("7ème mois", min_value=0 )



if st.button("Prédire"):
 forcast_demand = regr.predict(np.array([Mois_1, Mois_2, Mois_3, Mois_4, Mois_5, Mois_6, Mois_7 ]).reshape(1, -1))


 st.header("Résultats de prédiction")

#st.header("Résultats de prédiction")
 st.metric(label="Forcast demand", value= round(float(forcast_demand)), delta= 'units')

#months = ['Mois 1','Mois 2','Mois 3','Mois 4','Mois 5','Mois 6','Mois 7','Prediction']   
m = np.array([1,2,3,4,5,6,7,8])
A = np.array([Mois_1,Mois_2,Mois_3,Mois_4,Mois_5,Mois_6,Mois_7,forcast_demand])
fig = plt.figure() 
plt.plot(m, A, color = 'red')
plt.xlabel('Mois')
plt.ylabel('Demande')
plt.show()

fig_html = mpld3.fig_to_html(fig)
components.html(fig_html,height=500)

