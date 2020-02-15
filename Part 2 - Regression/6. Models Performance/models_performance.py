'''
# R Squared (R^2)

R Squared = 1 - (SSres / SStot)

Donde:
    - SSres = La suma del cuadrado de los errores de la linea de regresion
    - SStot = La suma del cuadrado de los errores de la recta que representa el promedio de los valores del set

Mientras la linea de regresion mas se asemeje a la tendencia de los datos y menor sea el SSres, mas cerca
de 1 resultara el R^2. Mientras mas cerca de 1, mas preciso es el modelo. Si el R^2 da 1, significa que no hay error.
El R^2 puede dar negativo, y seria el caso de que la linea de regresion tenga mas error que la linea del promedio,
es decir SSres > SStot.

BIGGER == BETTER


# Adjusted R Squared (Adj R^2)

En regresion lineal multiple, al agregar nuevas variables independientes al modelo para ver si mejora, el R^2 puede
aumentar (si efectivamente la variable mejora el modelo) o puede quedar igual que antes, pero nunca decrecer, ya que
al entrenarse el modelo, si la variable introducida empeora el modelo, su coeficiente terminara siendo 0, es decir,
el modelo la anula.
Por lo tanto, si al introducir una variable independiente que no tiene correlacion logica con la variable dependiente
pero por la distribucion de los datos o algun otro motivo mejora el R^2, no podemos saber si realmente mejora el
modelo o si es algo aleatorio.

Adj R^2 = 1 - (1-R^2) * ((n-1) / (n-p-1))

Donde:
    - p = cantidad de regresores (variables independientes)
    - n = tamaño de la muestra

El ultimo termino funciona como una penalizacion al agregar variables independientes que no ayudan al modelo.
El segundo termino decrece (bien) o se queda igual al agregar una variable independiente y el tercer termino
crece (mal), por lo que se crea un balance entre ambos. Por lo tanto para mejorar el Adj R^2, el aumento del R^2
tiene que ser lo suficientemente grande como para contrarrestar la penalizacion por introducir una nueva variable.


# Linear Regression Coefficients

Indican en que proporcion un cambio en una variable indepentiende, cambia la variable dependiente (en terminos de la
unidad en la que este expresada la variable independiente, es decir, no indica que una variable tenga mas impacto que
la otra, sino que tanto impacto tiene un cambio en una unidad de esa variable independiente sobre la variable
dependiente). El signo indica si al aumentar la variable independiente, aumenta la variable dependiente (+) o si
es al reves (-).
'''