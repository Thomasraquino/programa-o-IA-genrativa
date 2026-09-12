import streamlit as st    # interface grafica 
import pandas as pd       # tratamento de dados
from sklearn.linear_model import LinearRegression # o tipo de treinamento do modelo


st.header('PREVISÃO DE VENDAS🗓️')


dados_vendas = pd.DataFrame({


   'investimentos':[100,200,300,550,750,800],
   'faturamento':[1200,2500,3700,3900,5500,6900]


})


st.write(dados_vendas)


# treinar os dados 


X = dados_vendas[['investimentos']]
y = dados_vendas['faturamento']


model = LinearRegression().fit(X,y) # treina o modelo com os dados


investimento =  st.number_input('Digite o investimento', value = 150)


if investimento:
    if st.button('Analisar:'):
    
        previsao = model.predict([[investimento]])[0] #previsão
        st.write(f'Faturamento -  previsto R${previsao:.2f} **')# resultado

 




# analisar a previsão de vendas do mes de setembro