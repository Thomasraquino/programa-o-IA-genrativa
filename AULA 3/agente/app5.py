import streamlit as st

st.set_page_config(page_title="Atividade 5 - Sentimento Simples", page_icon="😊")

st.title("😊  Classificação de Sentimento Simples")
st.write("Classifica comentários em Positivo, Negativo ou Neutro com base em palavras-chave.")

comentario = st.text_area("Comentário do cliente:", value="O atendimento foi bom e o produto é muito bom")

if st.button("Classificar", type="primary"):
    positivas = ["bom", "excelente", "ótimo", "otimo", "maravilhoso"]
    negativas = ["ruim", "péssimo", "pessimo", "horrível", "horrivel"]
    
    texto_lower = comentario.lower()
    q_pos = sum(texto_lower.count(p) for p in positivas)
    q_neg = sum(texto_lower.count(p) for p in negativas)
    
    st.subheader("Resultado da Classificação:")
    if q_pos > q_neg:
        st.success("Sentimento: **Positivo** 😊")
    elif q_neg > q_pos:
        st.error("Sentimento: **Negativo** 😞")
    else:
        st.info("Sentimento: **Neutro** 😐")