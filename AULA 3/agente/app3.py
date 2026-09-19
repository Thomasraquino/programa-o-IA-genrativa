import streamlit as st

st.set_page_config(page_title="Atividade 3 - Detecção de Termos Negativos", page_icon="🚨")

st.title("🚨  Detecção de Palavras Negativas")
st.write("Verifica se a mensagem contém termos críticos para priorizar o atendimento.")

mensagem = st.text_area("Mensagem do cliente:", value="Olá, estou com um erro grave e o sistema está muito ruim")

if st.button("Verificar Mensagem", type="primary"):
    palavras_negativas = ["ruim", "péssimo", "pessimo", "erro", "defeito"]
    tem_negativa = any(termo in mensagem.lower() for termo in palavras_negativas)
    
    if tem_negativa:
        st.error("🚨 Prioridade Alta: Mensagem contém termos negativos!")
    else:
        st.success("✅ Prioridade Normal: Nenhuma palavra negativa detectada.")