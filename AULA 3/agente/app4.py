import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

st.set_page_config(page_title=" Remoção de Stopwords", page_icon="🧹")

@st.cache_resource
def carregar_nltk():
    nltk.download('punkt_tab')
    nltk.download('stopwords')

carregar_nltk()

st.title("🧹  Remoção de Stopwords em Português")
st.write("Filtra o texto removendo artigos, preposições e palavras irrelevantes.")

texto = st.text_area("Texto original:", value="O suporte para a conta do cliente foi excelente")

if st.button("Filtrar Texto", type="primary"):
    tokens = word_tokenize(texto.lower())
    stopwords_pt = set(stopwords.words('portuguese'))
    filtrado = [w for w in tokens if w not in stopwords_pt and w.isalpha()]
    
    st.subheader("Resultado:")
    st.write("**Tokens sem stopwords:**", filtrado)
    st.code(" ".join(filtrado))