import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
#se leen valores del csv con las columnas buscadas
dataframe =pd.read_csv('accidentes.csv', usecols=['Nivel inicial', 'Valor', 'Periodo Corto'])
dataframe.rename(columns={'Periodo Corto': 'anio'}, inplace=True)
#filtrar por los accidentes de transito con especifico valor
rslt_df = dataframe.loc[dataframe['Nivel inicial']== "Accidentes de tránsito terrestre en zonas urbanas y suburbanas  (Accidentes)"]
#rslt_df.rename(columns={'Periodo Corto': 'anio'}, inplace=True)
df=rslt_df.drop(columns=['Nivel inicial'])
df.describe()
#print(df)
X = df['anio'].values
Y = df['Valor'].values


X_train, X_test, y_train, y_test = train_test_split(
                                        X.reshape(-1,1),
                                        Y.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )

fig, ax = plt.subplots(figsize=(6, 3.84))

x = np.array(df['anio'].values).reshape((-1, 1))
y = np.array(df['Valor'].values)
# Create an instance of a linear regression model and fit it to the data with the fit() function:
model = LinearRegression().fit(x, y) 
# The following section will get results by interpreting the created instance: 
# Obtain the coefficient of determination by calling the model with the score() function, then print the coefficient:
r_sq = model.score(x, y)
print('Coeficiente de determinación:', r_sq)
# Print the Intercept:
print('intercept:', model.intercept_)
# Print the Slope:
print('slope:', model.coef_) 
# Predict a Response and print it:
y_pred = model.predict(x)
#print('Predicted response:', y_pred, sep='\n')
plt.scatter(x, y,linestyle='--',label="OLS",color='red')
plt.plot(x, y_pred, marker='o', label='Accidentes predichos')
plt.xlabel('Año')
plt.ylabel('No Accidentes')
plt.title('Regresión lineal para el no de accidentes en el estado de Sonora desde 1997 al 20202')
#plt.plot(x, predicciones, marker='+', label='Accidentes predichos1')
plt.show()
