import os

# Oculta avisos do sistema TensorFlow no terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

# 1. Preparação dos Dados
estudos = pd.DataFrame({
    'notas': [1, 2, 4, 6, 8, 10],
    'horas': [2, 4, 5, 7, 9, 10]
})

# Separando features (X) e target (y)
X = np.array(estudos['horas'], dtype=np.float32)
y = np.array(estudos['notas'], dtype=np.float32)

print("Dados de entrada (Horas):", X)
print("Dados de saída (Notas):", y)
print("-" * 40)

# 2. Criação do Modelo (Sintaxe Atualizada para Keras 3)
modelo = Sequential([
    Input(shape=(1,)),
    Dense(units=1)
])

# 3. Compilação do Modelo
modelo.compile(
    optimizer='sgd', 
    loss='mean_squared_error'
)

# Resumo do modelo
modelo.summary()
print("-" * 40)

# 4. Treinamento
print("Treinando o modelo...")
historico = modelo.fit(X, y, epochs=500, verbose=0)
print("Treinamento concluído!")
print("-" * 40)

# 5. Realizando Predições
horas_teste = np.array([[6.0]], dtype=np.float32)
previsao = modelo.predict(horas_teste, verbose=0)

print(f"Previsão para {horas_teste[0][0]} horas de estudo:")
print(f"Nota prevista: {previsao[0][0]:.2f}")