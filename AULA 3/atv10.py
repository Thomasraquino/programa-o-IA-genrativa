import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

herois = pd.DataFrame({
    'forca': [1, 2, 3, 7, 8, 10],
    'heroi': [0, 0, 0, 1, 1, 1]
})

X = herois[['forca']].to_numpy(dtype=np.float32)
y = herois['heroi'].to_numpy(dtype=np.float32)

modelo = Sequential([
    Input(shape=(1,)),
    Dense(units=1, activation='sigmoid')
])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

modelo.fit(X, y, epochs=500, verbose=0)

forca_teste = np.array([[5.0]], dtype=np.float32)
probabilidade = modelo.predict(forca_teste, verbose=0)[0][0]
classificacao = "Forte" if probabilidade >= 0.5 else "Fraco"

print(f"Força do herói: {forca_teste[0][0]}")
print(f"Chance de ser Forte: {probabilidade * 100:.1f}%")
print(f"Classificação: Herói {classificacao}")