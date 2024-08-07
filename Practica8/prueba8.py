# Vamos a hacer una red neuronal multicapa que pueda convertir grados Celsius a Fahrenheit
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Datos de entrenamiento 
celsius = np.array([0, -6.66667, -9.444444, 10, 18.3333, 22.2222, -15, 29.4444, 37.7778, -17.7778], dtype=float)
fahrenheit = np.array([32, 20, 15, 50, 65, 72, 5, 85, 100, 0], dtype=float)

# Construcción del modelo
capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

# Propiedades para el aprendizaje
modelo.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')

# Entrenamiento
print("Comenzando entrenamiento...")
historial = modelo.fit(fahrenheit, celsius, epochs=1000, verbose=False) #Entrenamos el modelo con fahrenheit para que nos de celsius
print("Modelo entrenado")

# Gráfica de los resultados de la función de pérdida
plt.xlabel("Época")
plt.ylabel("Magnitud de Pérdida")
plt.plot(historial.history["loss"])
plt.show()

# Predicciones y resultados
print("Predicciones")
resultado = modelo.predict(np.array([[100.0]]))  # Aquí se corrige la forma de la entrada
print("El resultado es " + str(resultado[0][0]) + " Fahrenheit")

print("Variables internas del modelo")
print(capa.get_weights()) #impresion de pesos internos (pesos y sesgo)