import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuração inicial da página
st.set_page_config(
    page_title="Validador de Modelos Preditivos",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧪 Validador de Modelos de Classificação")
st.caption("Protótipo interativo para teste de dados, treinamento e inferência em tempo real.")

# --- BARRA LATERAL: Configurações e Entrada de Dados ---
st.sidebar.header("1. Fonte de Dados")
data_source = st.sidebar.radio(
    "Escolha a origem dos dados:",
    ["Gerar Dados Sintéticos", "Carregar arquivo CSV"]
)

@st.cache_data
def generate_synthetic_data(samples, features, random_state):
    """Gera um conjunto de dados sintéticos para classificação binária."""
    X, y = make_classification(
        n_samples=samples,
        n_features=features,
        n_informative=max(2, features - 1),
        n_redundant=0,
        random_state=random_state
    )
    feature_names = [f"feature_{i+1}" for i in range(features)]
    df_x = pd.DataFrame(X, columns=feature_names)
    df_y = pd.Series(y, name="target")
    return pd.concat([df_x, df_y], axis=1), feature_names

df = None
features = []
target_col = "target"

if data_source == "Gerar Dados Sintéticos":
    n_samples = st.sidebar.slider("Número de amostras", 100, 2000, 500, step=100)
    n_features = st.sidebar.slider("Número de features", 2, 10, 4)
    seed = st.sidebar.number_input("Random Seed", value=42, step=1)
    df, features = generate_synthetic_data(n_samples, n_features, seed)
else:
    uploaded_file = st.sidebar.file_uploader("Upload do arquivo CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        target_col = st.sidebar.selectbox("Selecione a coluna alvo (target):", df.columns)
        features = [col for col in df.columns if col != target_col]

st.sidebar.markdown("---")
st.sidebar.header("2. Configuração do Modelo")
model_type = st.sidebar.selectbox(
    "Escolha o Algoritmo:",
    ["Random Forest", "Regressão Logística"]
)

test_size = st.sidebar.slider("Proporção do conjunto de teste (%)", 10, 50, 20) / 100.0

# --- CORPO PRINCIPAL DO APP ---
if df is not None:
    tab_data, tab_train, tab_predict = st.tabs([
        "📊 Dados", 
        "⚙️ Treinamento e Avaliação", 
        "🔮 Previsão em Tempo Real"
    ])

    # ABA 1: Exploração dos Dados
    with tab_data:
        st.subheader("Visualização dos Dados")
        st.write(f"**Dimensões:** {df.shape[0]} linhas × {df.shape[1]} colunas")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(df.head(10), use_container_width=True)
        with col2:
            st.markdown("**Estatísticas Descritivas:**")
            st.dataframe(df[features].describe().T[["mean", "std", "min", "max"]])

    # Processamento do Treinamento
    X = df[features]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    if model_type == "Random Forest":
        n_estimators = st.sidebar.slider("Número de Árvores (n_estimators)", 10, 200, 100, step=10)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    else:
        c_param = st.sidebar.number_input("Parâmetro C (Regularização)", value=1.0, min_value=0.01)
        model = LogisticRegression(C=c_param, max_iter=1000, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # ABA 2: Métricas do Modelo
    with tab_train:
        st.subheader("Desempenho do Modelo")
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Acurácia no Conjunto de Teste", f"{acc * 100:.2f}%")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**Matriz de Confusão:**")
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, index=["Real 0", "Real 1"], columns=["Pred 0", "Pred 1"])
            st.dataframe(cm_df)
            
        with col_m2:
            st.markdown("**Relatório de Classificação:**")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

    # ABA 3: Inferência Interativa
    with tab_predict:
        st.subheader("Entrada de Novos Dados")
        st.write("Ajuste os valores abaixo para realizar uma previsão em tempo real:")
        
        input_data = {}
        # Organiza os seletores de entrada em colunas dinâmicas
        cols = st.columns(min(len(features), 4))
        for idx, feature in enumerate(features):
            col = cols[idx % 4]
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            mean_val = float(df[feature].mean())
            
            input_data[feature] = col.number_input(
                label=feature,
                min_value=min_val,
                max_value=max_val,
                value=mean_val
            )
        
        input_df = pd.DataFrame([input_data])
        
        st.markdown("---")
        if st.button("Realizar Previsão", type="primary"):
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.success(f"**Classe Prevista:** `{prediction}`")
            with res_col2:
                prob_df = pd.DataFrame({
                    "Classe": model.classes_,
                    "Probabilidade": [f"{p*100:.2f}%" for p in probabilities]
                })
                st.dataframe(prob_df, hide_index=True)

else:
    st.info("Aguardando carregamento de dados via barra lateral para iniciar o aplicativo.")