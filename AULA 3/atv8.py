import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

musica = pd.DataFrame({
    'bpm': [80, 90, 100, 120, 140],
    'viral': [1, 2, 4, 7, 10]
})

X = musica[['bpm']].to_numpy(dtype=np.float32)
y = musica['viral'].to_numpy(dtype=np.float32)

modelo = Sequential([
    Input(shape=(1,)),
    Dense(units=1)
])

modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mse')
modelo.fit(X, y, epochs=500, verbose=0)

bpm_teste = np.array([[110.0]], dtype=np.float32)
previsao = modelo.predict(bpm_teste, verbose=0)

print(f"BPM da música: {bpm_teste[0][0]}")
print(f"Potencial de viralizar: {previsao[0][0]:.1f} / 10")