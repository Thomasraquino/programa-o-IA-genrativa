import streamlit as st
st.title('Painel de preferências')
st.selectbox("selecione uma opção", ["pyton","web"])
st.multiselect("múltiplas opções",["HTML","CSS","SQL","Git"])
st.button('Enviar')