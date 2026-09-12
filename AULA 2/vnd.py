import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Previsão de Vendas",
    page_icon="📊",
    layout="wide"
)

st.title("Sistema de previsão de vendas 🗃️")
st.write("Visualização de dados e previsão")

# Carregar os dados
dados = pd.read_csv("vendas.csv")

st.subheader("Dados de venda 🎲")
st.dataframe(dados, use_container_width=True)

# Separar dados para treinamento
x = dados[["mes"]]
y = dados["vendas"]

# Criar e treinar o modelo
modelo = LinearRegression()
modelo.fit(x, y)

# Escolher o mês
mes = st.number_input(
    "Digite o mês para previsão:",
    min_value=1,
    max_value=12,
    value=9
)

# Fazer a previsão usando o mês escolhido
previsao = modelo.predict([[mes]])[0]

st.subheader("Previsão")

st.success(
    f"Previsão de vendas para o mês {mes}: {previsao:.0f} vendas"
)
