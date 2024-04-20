import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import QuantileRegressor

#cargar el dataset
df =pd.read_csv('temp.csv', usecols=['Day', 'Month','Temp', 'Fecha', 'Day_year'])
datos=df['Temp']

#Maestra: debido a que los datos del datasrt no se ajustan tan facil a un RL
# Se utiliza el filtro IRQ para filtrar solo aquellos datos que se aproximen mas
# y esten menos dispersos, es decir, los que esten entre el segundo y tercer quartil
q1 = np.percentile(datos, 25)
q3 = np.percentile(datos, 75)
iqr = q3 - q1
# Definir el rango de tolerancia como .75 veces el IQR
rango_tolerancia =0.75 * iqr
# Calcular los límites inferior y superior
limite_inferior = q1 - rango_tolerancia
limite_superior = q3 + rango_tolerancia
print('lim. inferior:',limite_inferior)
print('lim. superior:',limite_superior)
# Filtrar los datos que caen dentro del rango de tolerancia
dataFrame = df.loc[(df["Temp"] >= limite_inferior) & (df["Temp"]  <= limite_superior)]
#print(dataFrame["Temp"])

X = np.array(dataFrame['Day_year']).reshape((-1, 1))  # Características: año y día 
y = np.array(dataFrame['Temp'])    # Etiquetas: temperatura

# Dividir los datos en conjuntos de entrenamiento y prueba, utilizando el 30% para prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# Crear un modelo de regresión lineal y ajustarlo a los datos de entrenamient
model = LinearRegression()
model.fit(X_train, y_train)
# Coeficientes de la regresión lineal
print('Coeficientes:', model.coef_)
print('Término independiente:', model.intercept_)
# Calcular la precisión del modelo en el conjunto de prueba
accuracy = model.score(X_test, y_test)
print('Precisión del modelo:', accuracy)

# Realizar predicciones
y_pred = model.predict(X_test)
# Visualización de los resultados
plt.scatter(X, y,color='red', label='Temp Registrada')
plt.xlabel('Dia del año')
plt.ylabel('Temperatura')
plt.title('Predicción de temperatura utilizando regresión lineal')
plt.plot(X_test, y_pred, marker='o', label='Dia del año')
plt.show()
