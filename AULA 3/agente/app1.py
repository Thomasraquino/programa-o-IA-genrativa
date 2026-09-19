import streamlit as st
import spacy

# Configuração da página Web
st.set_page_config(page_title="Analisador de Sentimento PT-BR", page_icon="🎭", layout="centered")

# Carrega o modelo de Língua Portuguesa do spaCy
@st.cache_resource
def carregar_spacy():
    return spacy.load("pt_core_news_sm")

nlp = carregar_spacy()

# Dicionário léxico simples para análise de sentimento em português
PALAVRAS_POSITIVAS = {
    "excelente", "ótimo", "otimo", "bom", "maravilhoso", "adoro", "gostei", 
    "incrivel", "incrível", "superou", "satisfeito", "feliz", "perfeito", "amor"
}

PALAVRAS_NEGATIVAS = {
    "péssimo", "pessimo", "ruim", "horrível", "horrivel", "odeio", "triste", 
    "decepcionado", "decepção", "decepcao", "pior", "lixo", "problema", "erro"
}

def analisar_sentimento_pt(doc):
    pos_count = 0
    neg_count = 0
    total_palavras = 0

    for token in doc:
        palavra = token.text.lower()
        if palavra in PALAVRAS_POSITIVAS:
            pos_count += 1
        elif palavra in PALAVRAS_NEGATIVAS:
            neg_count += 1
        if token.is_alpha:
            total_palavras += 1

    if total_palavras == 0:
        return {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0}

    score_pos = pos_count / total_palavras
    score_neg = neg_count / total_palavras
    score_neu = 1.0 - (score_pos + score_neg)
    compound = score_pos - score_neg

    return {"pos": score_pos, "neg": score_neg, "neu": score_neu, "compound": compound}

# Interface do Usuário
st.title("🎭 Analisador de Sentimento em Português")
st.write("Digite uma frase em português para analisar a polaridade do sentimento e a estrutura gramatical.")

# Caixa de texto para o usuário
frase = st.text_input("Frase para teste:", value="O produto é excelente e superou todas as minhas expectativas!")

if st.button("Analisar", type="primary"):
    if frase.strip():
        # Processamento de texto com spaCy
        doc = nlp(frase)
        
        # Tokens
        tokens = [token.text for token in doc]
        
        # Análise de sentimento
        pontos = analisar_sentimento_pt(doc)
        
        st.divider()
        
        # Exibição das métricas de sentimento
        st.subheader("📊 Análise de Sentimento")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Negativo", f"{pontos['neg']:.2%}")
        col2.metric("Neutro", f"{pontos['neu']:.2%}")
        col3.metric("Positivo", f"{pontos['pos']:.2%}")
        col4.metric("Pontuação Geral", f"{pontos['compound']:.2f}")
        
        # Classificação geral baseada no compound
        if pontos['compound'] > 0:
            st.success("Sentimento Geral: **Positivo** 😊")
        elif pontos['compound'] < 0:
            st.error("Sentimento Geral: **Negativo** 😞")
        else:
            st.info("Sentimento Geral: **Neutro** 😐")
            
        st.divider()
        
        # Exibição dos Tokens e Classes Gramaticais
        st.subheader("🔤 Processamento de Texto")
        st.write("**Tokens (Palavras fatiadas):**", tokens)
        
        # Tabela com as classes gramaticais em português
        st.write("**Etiquetagem Gramatical (POS Tags em Português):**")
        dados_gramaticais = [
            {"Palavra": token.text, "Classe Gramatical (POS)": token.pos_, "Explicação": spacy.explain(token.pos_)}
            for token in doc
        ]
        st.dataframe(dados_gramaticais, use_container_width=True)
    else:
        st.warning("Por favor, digite alguma frase antes de analisar.")

        