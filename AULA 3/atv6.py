import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

filmes = pd.DataFrame({
    'duracao': [80, 90, 100, 110, 120],
    'nota': [4, 5, 7, 8, 9]
})

X = filmes[['duracao']].to_numpy(dtype=np.float32)
y = filmes['nota'].to_numpy(dtype=np.float32)

modelo = Sequential([
    Input(shape=(1,)),
    Dense(units=1)
])

modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mse')
modelo.fit(X, y, epochs=500, verbose=0)

duracao_teste = np.array([[105.0]], dtype=np.float32)
previsao = modelo.predict(duracao_teste, verbose=0)

print(f"Duração do filme: {duracao_teste[0][0]} min")
print(f"Nota estimada: {previsao[0][0]:.1f}")