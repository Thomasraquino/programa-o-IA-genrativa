import streamlit as st
import nltk
from nltk.tokenize import word_tokenize

# Configuração da página
st.set_page_config(page_title="Atividade 2 - Frequência de Palavras", page_icon="📊")

# Download automático do recurso necessário do NLTK
@st.cache_resource
def carregar_nltk():
    nltk.download('punkt_tab')

carregar_nltk()

st.title("📊  Frequência de Palavras")
st.write("Insira um texto abaixo para analisar a contagem e frequência de cada palavra.")

# Entrada de texto do usuário
texto = st.text_area(
    "Texto para análise:", 
    value="o produto é bom o produto é excelente o serviço é bom",
    height=150
)

if st.button("Calcular Frequência", type="primary"):
    if texto.strip():
        # Tokenização e contagem de frequência
        palavras = word_tokenize(texto.lower())
        frequencia = nltk.FreqDist(palavras)
        
        st.subheader("Resultados:")
        
        # Converte o dicionário de frequência em uma lista para exibir em tabela
        dados = [{"Palavra / Token": palavra, "Quantidade": qtd} for palavra, qtd in frequencia.items()]
        
        # Exibe a tabela interativa na tela do Streamlit
        st.dataframe(dados, use_container_width=True)
    else:
        st.warning("Por favor, digite um texto antes de analisar.")