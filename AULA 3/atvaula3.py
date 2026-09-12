from sklearn.tree import DecisionTreeClassifier
import numpy as np
import streamlit as st

# Dados de treinamento
x = np.array([
    [1, 0],
    [3, 3],
    [9, 3],
    [9, 2],
    [7, 1]
])

y = np.array([0, 1, 0, 1, 0])

# Criando e treinando o modelo
modelo = DecisionTreeClassifier()
modelo.fit(x, y)

# Interface
st.title("Classificação de clientes🧐")

uso = st.number_input(
    "Quantidade de vezes que o produto foi utilizado",
    min_value=0,
    step=1
)

reclamacoes = st.number_input(
    "Quantidade de reclamações",
    min_value=0,
    step=1
)

st.title('análise')

if st.button("Classificar"):

    previsao = modelo.predict([[uso, reclamacoes]])[0]

    if previsao == 0:
        st.write("Cliente consolidado")
    else:
        st.write("Cliente não consolidado")
