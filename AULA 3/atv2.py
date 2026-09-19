import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


# 1. PREPARAÇÃO DOS DADOS


# Criação do DataFrame original fornecido
estudos = pd.DataFrame(
    {"notas": [1, 2, 4, 6, 8, 10], "horas": [2, 4, 5, 7, 9, 10]}
)

# Definimos X  y 
# Convertemos para arrays do NumPy com tipo float32, que é o padrão exigido pelo TensorFlow
X = estudos["horas"].to_numpy(dtype=np.float32)
y = estudos["notas"].to_numpy(dtype=np.float32)


# 2. CONSTRUÇÃO DA REDE NEURAL (MODELO)


# O Sequential é uma pilha linear de camadas de neurônios
modelo = Sequential(
    [
        # Uma única camada Densa (Dense) com 1 neurônio e 1 entrada.
        # Em redes neurais, 1 neurônio sem função de ativação equivale matematicamente
        # a uma equação de Regressão Linear Simples: y = w * x + b
        Dense(units=1, input_shape=[1])
    ]
)

# COMPILAÇÃO DO MODELO

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss="mse"
)

# ==============================================================================
# 4. TREINAMENTO (FIT)
# ==============================================================================

print("Iniciando o treinamento do modelo TensorFlow...")

# 'epochs' é o número de vezes que a rede neural vai olhar todo o conjunto de dados para ajustar os pesos
historico = modelo.fit(X, y, epochs=500, verbose=0)

print("Treinamento concluído!\n")

# ==============================================================================
# 5. PREVISÃO E TESTE
# ==============================================================================

# Vamos testar a rede neural com uma pessoa que estudou 8 horas
horas_teste = np.array([8.0], dtype=np.float32)

# Fazemos a predição usando o modelo treinado
nota_prevista = modelo.predict(horas_teste, verbose=0)

# Exibimos os resultados formatados
print("=" * 45)
print("     🎓 IA DAS NOTAS ESCOLARES (TENSORFLOW) 🎓")
print("=" * 45)
print(f"Horas de estudo de teste: {horas_teste[0]}h")
print(f"Nota prevista pelo modelo: {nota_prevista[0][0]:.2f}")
print("=" * 45)