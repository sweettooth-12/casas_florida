# El presente analisis tiene como objetivo encontrar oportunidades de compra de casas.
# 
# - Para ello se utilizaran precios de casas en algunos counties de Florida, especificamente los cercanos a Miami tomadas de la web de Zillow a mediados de septiembre de 2024.
# - Una oportunidad de compra la definiremos como aquella situación en la cual una casa tiene un precio que puede ser menor al promedio de casas similares o con características similares.
# 
# Importamos las librerias que utilizaremos para el analisis

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from statsmodels.formula.api import ols
from scipy import stats
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

# Veremos las columnas y los principales valores del archivo de precios de casas de Florida
# Cada renglón en este data set es una propiedad

casas = pd.read_excel('FLORIDArevisionfeatures.xlsx')

casas.head()

casas.tail()

# Número de datos

casas.shape

# Nombres de las columnas

casas.columns

# Analicemos rapidamente los principales valores de agregacion con la funcion describe de python

casas.describe()

# Veamos el numero y tipo de datos

casas.info()

# Verificamos los valores nulos dentro de las columnas de nuestro interes

casas['price'].isnull().sum()

casas['bedrooms'].isnull().sum()

casas['bathrooms'].isnull().sum()

casas['latitude_InfoTOD'].isnull().sum()

casas['longitude_InfoTOD'].isnull().sum()

casas['yearBuilt'].isnull().sum()

casas['distancenerabSchool'].isnull().sum()

casas['homeStatus'].isnull().sum()

# Nos deshacemos de los valores nulos dentro de las columnas de nuestro interes

casas = casas.dropna(subset=['bedrooms', 'bathrooms', 'yearBuilt'])

casas.info()

# Ocuparemos los valores del segundo al tercer cuartil y los estatus de casa que estan en venta o relacionados a ella

casas2 = casas[( (casas['price'] > 389000) & (casas['price'] < 1300000) ) & \
( (casas['bathrooms'] > 1) & (casas['bathrooms'] < 4) ) & \
( (casas['bedrooms'] > 1) & (casas['bedrooms'] < 4) )  & \
( (casas['homeStatus']== 'for_sale')|(casas['homeStatus']== 'pre_forclosure')|(casas['homeStatus']== 'recently_sold')|(casas['homeStatus']== 'sold')) ]

# Tras la limpieza de los datos, observemos como ha cambiado el numero de entradas para el analisis

casas.shape, casas2.shape

# Veamos las graficas de distribucion de las columnas de nuestro interes

plt.clf()
casas2[['price', 'bedrooms', 'bathrooms', 'latitude_InfoTOD', 'longitude_InfoTOD', 'yearBuilt']].hist(bins = 50, figsize = (20, 15))

plt.show()

# Formulamos la regresion

formula = 'price ~ bedrooms + bathrooms + latitude_InfoTOD + longitude_InfoTOD + yearBuilt'

results = ols(formula, casas2).fit()

print(results.summary())

# Definimos la matriz X y el vector Y

X = casas2[["bedrooms", "bathrooms", "latitude_InfoTOD", "longitude_InfoTOD", 'yearBuilt']]

y = casas2["price"].values.reshape(-1, 1)

X.shape, y.shape

# Separamos los datos en prueba y entrenamiento

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.25)

X_train.shape, X_test.shape, 6322/(6322+2108)

# Se observa que la separacion de datos en prueba y entrenamiento es correcta por su valor de 0.749 para
# el set de entrenamiento.

# Dado que los datos de los coeficientes son muy dispares entre si, usaremos la herramienta de
# escalamiento.

X_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)

# Transformamos los datos de entrenamiento y prueba usando X_scaler y y_scaler

X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
y_train_scaled = y_scaler.transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# Creamos el modelo de regresion lineal y lo ajustamos a los datos ya escalados

model = LinearRegression()
model.fit(X_train_scaled, y_train_scaled)

# Extraemos los coeficientes de la regresion

model.intercept_, model.coef_

# Gracias al escalamiento los coeficientes son mas manejables para el analisis.

# Creamos las predicciones

predictions = model.predict(X_test_scaled)
predictions

# Graficamos los resultados

# PLOT

plt.clf()

plt.scatter( model.predict(X_train_scaled), model.predict(X_train_scaled) - y_train_scaled,
             c = "darkblue", label = "Training Data" )

plt.scatter( model.predict(X_test_scaled), model.predict(X_test_scaled) - y_test_scaled,
             c = "orange", label = "Testing Data" )

plt.legend()

plt.hlines( y = 0, xmin = y_test_scaled.min(), xmax = y_test_scaled.max() )

plt.title("Residual Plot")

plt.show()

# PLOT

plt.clf()
plt.scatter( model.predict(X_train_scaled), y_train_scaled,
             c = "darkblue", label = "Training Data" )

plt.scatter( model.predict(X_test_scaled), y_test_scaled,
             c = "orange", label = "Testing Data" )

plt.legend()

plt.title("Predicted Plot")

plt.show()

# Graficamos los datos escalados

plt.scatter( y_scaler.inverse_transform(model.predict(X_train_scaled)),
             y_scaler.inverse_transform(y_train_scaled),
             c = "darkblue", label = "Training Data" )

plt.scatter( y_scaler.inverse_transform(model.predict(X_test_scaled)),
             y_scaler.inverse_transform(y_test_scaled),
             c = "orange", label = "Testing Data" )

plt.legend()

plt.title("Predicted Plot")

plt.show()

# Comparamos R2 entre el set de entrenamiento y el set de evaluacion

model.score(X_train_scaled, y_train_scaled), model.score(X_test_scaled, y_test_scaled)

# El modelo no sufre de sobreajuste porque el R2 del set de evaluacion es superior al del set de entrenamiento.