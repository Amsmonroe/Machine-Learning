#Aquí haremos la segunda red neuronal
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Datos
celsius = np.array([0, -6.66667, -9.444444, 10, 18.3333, 22.2222, -15, 29.4444, 37.7778, -17.7778], dtype=float)
fahrenheit = np.array([32, 20, 15, 50, 65, 72, 5, 85, 100, 0], dtype=float)

oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1]) # Capa oculta
oculta2 = tf.keras.layers.Dense(units=3) # Capa oculta
salida = tf.keras.layers.Dense(units=1) # Capa de salida
modelo = tf.keras.Sequential([oculta1, oculta2, salida])

# Propiedades para el aprendizaje
modelo.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')

# Entrenamiento
print("Comenzando entrenamiento...")
historial = modelo.fit(fahrenheit, celsius, epochs=1000, verbose=False) #Entrenamos el modelo con fahrenheit para que nos de celsius
print("Modelo entrenado")

# Predicciones y resultados
plt.xlabel("Época")
plt.ylabel("Magnitud de Pérdida")
plt.plot(historial.history["loss"])

print("Predicciones")
resultado = modelo.predict(np.array([[100.0]]))  # Aquí se corrige la forma de la entrada
print("El resultado es " + str(resultado[0][0]) + " Fahrenheit")

print("Variables internas del modelo")
print(oculta1.get_weights())
print(oculta2.get_weights())
print(salida.get_weights())