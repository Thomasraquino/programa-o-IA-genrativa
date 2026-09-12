"""
Aplicação Web de Previsão de Vendas com Streamlit e TensorFlow.
Autor: Especialista ML & Python
Descrição: Interface interativa para análise exploratória de vendas e
            previsão de séries temporais via Rede Neural Keras/TensorFlow.
"""

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Configuração da página Streamlit
st.set_page_config(
    page_title="Previsão de Vendas com TensorFlow",
    page_icon="📈",
    layout="wide"
)


@st.cache_data
def carregar_dados_padrao() -> pd.DataFrame:
    """Carrega dataset inicial via dicionário Python."""
    dados_vendas = {
        "Data": pd.date_range(start="2026-01-01", periods=12, freq="M"),
        "Vendas_Unidades": [120, 135, 150, 160, 190, 210, 230, 250, 280, 300, 310, 340],
        "Investimento_Mkt": [10, 12, 15, 14, 18, 20, 22, 25, 27, 30, 31, 35]
    }
    return pd.DataFrame(dados_vendas)


def preparar_dados(vendas: np.ndarray, janela: int):
    """Prepara as sequências de entrada (X) e alvo (y) para o modelo."""
    X, y = [], []
    for i in range(len(vendas) - janela):
        X.append(vendas[i : i + janela])
        y.append(vendas[i + janela])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def treinar_modelo_tensorflow(X: np.ndarray, y: np.ndarray, epocas: int) -> tf.keras.Model:
    """Compila e treina uma rede neural de regressão."""
    modelo = Sequential([
        Dense(16, activation='relu', input_shape=(X.shape[1],)),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    
    modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])
    modelo.fit(X, y, epochs=epocas, verbose=0)
    return modelo


def main():
    st.title("📈 Dashboard de Previsão de Vendas com IA")
    st.write("Aplicação interativa para análise exploratória e previsão via TensorFlow.")