import os

# Oculta mensagens informativas e avisos do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

sorvete = pd.DataFrame(
    {"temperatura": [18, 20, 24, 27, 30, 35], "vendas": [20, 25, 40, 55, 70, 100]}
)

X = sorvete["temperatura"].to_numpy(dtype=np.float32)
y = sorvete["vendas"].to_numpy(dtype=np.float32)

modelo = Sequential([Input(shape=(1,)), Dense(units=1)])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss="mse"
)

modelo.fit(X, y, epochs=1000, verbose=0)

temperatura_teste = np.array([[32.0]], dtype=np.float32)
vendas_previstas = modelo.predict(temperatura_teste, verbose=0)

print(f"Temperatura de teste: {temperatura_teste[0][0]}°C")
print(f"Previsão de sorvetes vendidos: {vendas_previstas[0][0]:.1f} unidades")