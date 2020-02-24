'''
# Confussion Matrix

False Positive (FP): se predijo como verdadero pero en realidad es falso
False Negative (FN): se predijo como falso pero en realidad es verdadero

Estos valores son los que se observan en la diagonal (0;1) (1;0) de la Confussion Matrix

En la confussion matrix, las filas representan los valores reales, y las columnas las predicciones
Ejemplo:

            y_pred
            0     1
y_actual 0  35    FP
         1  FN    50

Rates:
    - Accuracy Rate: Predicciones Correctas / Total de Predicciones
    - Error Rate: Predicciones Incorrectas / Total de Predicciones


# Accuracy Paradox

Si para un modelo, calculo el Accuracy Rate, me da un cierto porcentaje de precision. Si despues agarro, y hago que el
modelo prediga que todas las muestras pertenencen a la misma clase, una de las columnas de la matriz de confusiones
quedaria en cero, y al calcular nuevamente el Accuracy Rate, podria darme incluso mas alto que antes, siendo que el
comportamiento del modelo no logico. Esto muestra que no hay que basarse unicamente en el Accuracy Rate para medir
la precision del modelo, no es un buen indicador.


# CAP Curve (Cumulative Accuracy Profile Curve) --> ver video "CAP Curve"

Nos permite ver que tan bueno es nuestro modelo
'''