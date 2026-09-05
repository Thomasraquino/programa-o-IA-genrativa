

import streamlit as st

import streamlit as st
import pandas as pd

# procoding -> não utiliza IA generativa

dados = pd.read_csv('vendas.csv')

st.header(' calculadora STREAMLIT 🗿​ ' )
st.write('ADICIONE O NÚMEROS PARA CALCULAR')

n1  = st.number_input('digite um número1')
n2  = st.number_input('digite um número2: ', value= 0.0)

soma_, div_, sub_, mult_ = st.columns(4)

if  soma_.button('+'):
    soma = n1 + n2
    st.info(soma) 
elif sub_.button('-'):
     sub = n1 - n2
     st.info(sub)
elif mult_.button('x'):
     mult= n1 + n2
     st.info(mult)
elif div_.button(':'):
    div = n1 / n2
    st.info(div)

st.map()

st.header('ANÁLISE DE DADOS')

st.table(dados)

st.bar_chart(dados, x = 'ano' ,y = 'lucro')
st.scatter_chart(dados, x = 'venda' , y = 'lucro' )
st.line_chart(dados, x = 'ano' , y = 'venda')

