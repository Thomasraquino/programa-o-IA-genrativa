import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

alunos = pd.DataFrame({
    'faltas': [0, 1, 2, 5, 7, 10],
    'resultado': [1, 1, 1, 0, 0, 0]
})

X = alunos[['faltas']].to_numpy(dtype=np.float32)
y = alunos['resultado'].to_numpy(dtype=np.float32)

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

faltas_teste = np.array([[3.0]], dtype=np.float32)
probabilidade = modelo.predict(faltas_teste, verbose=0)[0][0]
aprovado = 1 if probabilidade >= 0.5 else 0

print(f"Faltas do aluno: {faltas_teste[0][0]}")
print(f"Probabilidade de Aprovação: {probabilidade * 100:.1f}%")
print(f"Classificação Final: {'Aprovado' if aprovado == 1 else 'Reprovado'}")