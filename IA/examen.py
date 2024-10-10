import tensorflow  as tf
import numpy as np

variable =  np.array([(2(x))^-2,(4(x))^-7,(2(x)^2,(4(x)^3)) ],  dtype=float) # type: ignore
Derivada = np.array([(4/((x)^-3)), (21/((x)^-8)), 4(x), 12((x)^2)],  dtype=float) # type: ignore

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo=  tf.keras.models.Sequential([capa])

modelo.compile(
    optimizer= tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Comenzando entrenamiento...")
historial = modelo.fit(variable, Derivada, epochs=1000,  verbose=False)
print("Modelo entrenado")

import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdidas")
plt.plot(historial.history["loss"])
plt.show()

print("Hagamos una predicción!")
resultado = modelo.predict(np.array([[12((x)^4)]])) # type: ignore
print("El resultado es " + str(resultado) + "Derivada")