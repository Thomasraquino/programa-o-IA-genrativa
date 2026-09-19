import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

pets = pd.DataFrame({
    'passeios': [1, 2, 3, 4, 5],
    'felicidade': [2, 4, 5, 8, 10]
})

X = pets[['passeios']].to_numpy(dtype=np.float32)
y = pets['felicidade'].to_numpy(dtype=np.float32)

modelo = Sequential([
    Input(shape=(1,)),
    Dense(units=1)
])

modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mse')
modelo.fit(X, y, epochs=500, verbose=0)

passeios_teste = np.array([[3.5]], dtype=np.float32)
previsao = modelo.predict(passeios_teste, verbose=0)

print(f"Passeios: {passeios_teste[0][0]}")
print(f"Felicidade estimada: {previsao[0][0]:.1f} / 10")