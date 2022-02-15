# Potenciales interátomicos de aprendizaje automático

**Behler2016**: Las simulaciones computacionales cada vez son más escenciales en 
física, química, ciencias de los materiales, etc. Para mantenerse en el estado 
del arte con los experimentos y la complejidad en aumento de los sistemas 
investigados es necesario que las simulaciones sean más realistas, es decir, 
que se puedan simular a escalas más grandes sin perder precisión. Los 
potenciales de Machine Learning ofrecen una alternativa para la representación 
de la superficie energía-potencial, entrenando en datos que provienen de 
calculos de la estructura electrónica. 

**Mueller2020**: (Actualización a Behler2016). Acelerar simulaciones en la escala 
atómica perdiendo poco en precisión.

## Introducción

**Behler2016**:
+ Definiciones de ML.
+ DFT PES a MD, definición de potencial interatómico.
+ eficiencia (MD) vs precisión (DFT): ML toma ventajas de ambos.
+ Definción de potencial de ML (3 puntos, discutido en el parrafo anterior).

**Mueller2020**:
+ Aproximación de Born-Oppenheimer para explicar la PES y que se puede obtener
una vez que se la conoce.
+ Solución a la eq de Schrödinger -> aproximación DFT (más usada).
+ Ejemplos de potenciales interatómicos previos: Coulomb, LJ, EAM, Tersoff,
MEAM, etc. ML para optimizar estos parametros (sólo mencionar).
+ Aprendizaje automático supervisado (remarca). y = f(x), donde x=configuraciones,
y=PES, f=potencial interatómico. Tres pasos:
    - espacio de hipotesis,
    - función objetivo (el error cuadrático de la energía, fuerzas, virial, con 
    respecto a los datos de entrenamiento),
    - la determinación de un método para buscar en el espacio de hipótesis espacio 
    de hipótesis para las buenas funciones.
+ _constrains_ físicos:
    - interacción de corto alcance, se interactúa más con los átomos que están 
    más cerca,
    - invariancia frente a simetrías,
    - varía suavemente.

## Descriptores

**Behler2016**:
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

**Behler2016**:
+ _Redes neuronales_: Primero para entender el cerebro. El primero en usarse 
(1995). Esquema feed-forward. Optimizado en un proceso iterativo basado en el
gradiente. Se desprende _Deep Learning_ (todavía no del todo desarrollado 
cuando se escribió la perspectiva, agregar directamente después). 
+ _Potenciales de aproximación gaussiana y métodos kernel_: Combina algún
descriptor con un kernel que relaciona la estructura con la energía. Es un 
suma pesada sobre las energías conocidas de los entornos del conjunto de 
entrenamiento. Combinación lineal de funciones que dependen del descriptor.
_Squared exponential kernel_.
+ _Support vector machines_: ¿rarely used here? Fig 3. Support vector         
regression.
+ _Spectral neighbor analysis potential_: versión lineal de GAPs.

**Mueller2020**:
+ _Moment tensor potentials_: Combinación lineal en la base de funciones 
polinomiales de interacciones de 1-cuerpo, 2-cuerpos y 3-cuerpos. Los parametros
se fitean usando cuadrados mínimos.
+ _Message-passing networks_: _graph networks_ donde los nodos de la red son los
átomos y los átomos cercanos entre sí son conectados por aristas. La información
se pasa iterativamente de un nodo a sus vecinos. Vectores de características son
asignatos a cada nodo y arista del grafo. Cada iteración agrega un órden de 
vecino nuevo, es decir para _n_ iteraciónes, las features tienen información de 
los _n_ vecinos cercanos. Pueden ser usados para generar potenciales interatómicos
pero no es lo usual.
+ _Symbolic regression_: Usar la computador para buscar dentro de las expresiones
matemáticas simples cual representa mejor la PES, usando programación genética, 
en la cual se optimiza tanto los parámetros como la forma funcional (es posible,
si se usan datos de LJ, recuperar LJ). Una gran ventaja es la simplicidad de los
modelos. Hay poca investigación en esta área, sólo se probó en Cu y no se sabe
si funcionará para dos o más elementos a la vez.

## Conclusiones

**Behler2016**:
+ Ventajas:
    - Cómputo rápido a la hora de producción.
    - Energías precisas, cercanas a las de los métodos de estructura 
    electrónica. 
+ Desventajas:
    - Se necesitan muchos datos de entrenamientos y es costoso generarlos.
    - Requieren mucho test y validación. Los potenciales de Dinámica Molecular
    pueden fallar tremendamente si no se los construye de apropiadamente.

**Mueller2020**:
