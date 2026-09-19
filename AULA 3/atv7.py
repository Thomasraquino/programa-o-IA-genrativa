import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

pizza = pd.DataFrame({
    'tamanho': [20, 25, 30, 35, 40],
    'preco': [20, 30, 40, 50, 60]
})

X = pizza[['tamanho']].to_numpy(dtype=np.float32)
y = pizza['preco'].to_numpy(dtype=np.float32)

modelo = Sequential([
    Input(shape=(1,)),
    Dense(units=1)
])

modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mse')
modelo.fit(X, y, epochs=500, verbose=0)

tamanho_teste = np.array([[32.0]], dtype=np.float32)
previsao = modelo.predict(tamanho_teste, verbose=0)

print(f"Tamanho da pizza: {tamanho_teste[0][0]} cm")
print(f"Preço estimado: R$ {previsao[0][0]:.2f}")