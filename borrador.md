# Potenciales interátomicos de aprendizaje automático

_Behler2016_: Las simulaciones computacionales cada vez son más escenciales en 
física, química, ciencias de los materiales, etc. Para mantenerse en el estado 
del arte con los experimentos y la complejidad en aumento de los sistemas 
investigados es necesario que las simulaciones sean más realistas, es decir, 
que se puedan simular a escalas más grandes sin perder precisión. Los 
potenciales de Machine Learning ofrecen una alternativa para la representación 
de la superficie energía-potencial, entrenando en datos que provienen de 
calculos de la estructura electrónica. 

## Introducción

_Behler2016_:
+ Definiciones de ML.
+ DFT PES a MD, definición de potencial interatómico.
+ eficiencia (MD) vs precisión (DFT): ML toma ventajas de ambos.
+ Definción de potencial de ML (3 puntos, discutido en el parrafo anterior).

## Descriptores

_Behler2016_:
+ Descriptor definido a partir de las configuraciones de los átomos.
+ Requerimientos del descriptor: translación, rotacion, permutación...
simetrías. Independiente del tamaño del problema. Rápido de calcular y 
diferenciable.
+ Algunos descriptores:
    - Funciones de simetría centradas en el átomo: Entorno químico, posiciones
    de los átomos hasta cierto radio de corte. Función de corte. Función
    radial y función angular. Función radial y angular centradas en el par.
    - Biespectro de la densidad de vecinos: Expansión del entorno en serie 
    de harmónicos esferícos. Matriz del biespectro.
    - Superposición suave de las posiciones atómicas: Gaussianas en vez de
    funciones delta como en el caso anterior. Hay que incorporar la 
    invariancia rotacional "a mano".
    - Matriz de Coulomb: Autovalores de la matriz (relacionada a la distancia).

## Potenciales de ML

_Behler2016_:
+ Redes neuronales: Primero para entender el cerebro. El primero en usarse 
(1995). Esquema feed-forward. Optimizado en un proceso iterativo basado en el
gradiente. Se desprende _Deep Learning_ (todavía no del todo desarrollado 
cuando se escribió la perspectiva, agregar directamente después). 
+ Potenciales de aproximación gaussiana y métodos kernel: Combina algún
descriptor con un kernel que relaciona la estructura con la energía. Es un 
suma pesada sobre las energías conocidas de los entornos del conjunto de 
entrenamiento. Combinación lineal de funciones que dependen del descriptor.
_Squared exponential kernel_.
+ _Support vector machines_: ¿rarely used here? Fig 3. Support vector         
regression.
+ _Spectral neighbor analysis potential_: versión lineal de GAPs.

## Conclusiones

_Behler2016_:
+ Ventajas:
    - Cómputo rápido a la hora de producción.
    - Energías precisas, cercanas a las de los métodos de estructura 
    electrónica. 
+ Desventajas:
    - Se necesitan muchos datos de entrenamientos y es costoso generarlos.
    - Requieren mucho test y validación. Los potenciales de Dinámica Molecular
    pueden fallar tremendamente si no se los construye de apropiadamente.
