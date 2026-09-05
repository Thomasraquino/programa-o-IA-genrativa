import streamlit as st
st.title('Formulário cadastral📋')
nome=st.text_input('Nome')
idade=st.number_input('Idade', value= 0)
st.checkbox('Declaro que aceito os termos de uso')


if st.button('enviar'):
    st.write(nome)
    st.write(idade)
