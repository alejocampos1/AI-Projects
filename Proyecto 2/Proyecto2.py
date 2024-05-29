# Importar librerias

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
url = "https://raw.githubusercontent.com/alejocampos1/AI-Projects/main/Proyecto%202/datos_de_ventas.csv"
sales_df = pd.read_csv(url)

# Visualizar los datos
sns.scatterplot(data=sales_df, x='Temperature', y='Revenue')

# Dataset para entrenamiento
x_train = sales_df['Temperature']
y_train = sales_df['Revenue']

# Crear el modelo
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

# Resumen del modelo
model.summary()

# Compilar el modelo
model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')

# Entrenar el modelo
history = model.fit(x_train, y_train, epochs=1000)

# Visualizar los resultados
keys = history.history.keys()

# Graficar el entrenamiento
plt.plot(history.history['loss'])
plt.title('Progreso del entrenamiento')
plt.xlabel('Epoch')
plt.ylabel('Pérdida')
plt.legend(['Pérdida'])

# Obtener pesos
weights = model.get_weights()

# Predicción
temp = 35
revenue = model.predict(np.array([temp])).item()
print('La ganancia según la red neuronal es de:', int(revenue))

# Visualizar la predicción
plt.scatter(x_train, y_train, color='gray', label='Datos')
plt.plot(x_train, model.predict(x_train), color='red', label='Predicción')
plt.title('Predicción de ganancias')
plt.xlabel('Temperatura')
plt.ylabel('Ganancia')
plt.legend()
plt.show()
