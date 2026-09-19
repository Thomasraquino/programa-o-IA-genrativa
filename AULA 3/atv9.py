import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

cafe = pd.DataFrame({
    'xicaras': [1, 2, 3, 4, 5],
    'energia': [2, 4, 6, 8, 10]
})

X = cafe[['xicaras']].to_numpy(dtype=np.float32)
y = cafe['energia'].to_numpy(dtype=np.float32)

modelo = Sequential([
    Input(shape=(1,)),
    Dense(units=1)
])

modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mse')
modelo.fit(X, y, epochs=500, verbose=0)

xicaras_teste = np.array([[3.0]], dtype=np.float32)
previsao = modelo.predict(xicaras_teste, verbose=0)

print(f"Xícaras de café: {xicaras_teste[0][0]}")
print(f"Nível de energia: {previsao[0][0]:.1f} / 10")