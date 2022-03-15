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

**Deringer2020**: Los métodos de mecánica cuántica ofrecen gren precisión y poder
predictivo a partir de simulaciones en escalas atómicas, pero rápidamente alcanzan
su límite cuando los sistemas electroquímicos se hacen más complejos, por ejemplo 
cuando se estudian fases amorfas o cuando las reacciones que se quieren estudiar
se encuentran en interfases entre electrodo y electrolito.

**Mishin2021**: Mientras que los potenciales tradicionales se derivan de 
conocimientos físicos, los potenciales de ML usan regresiones matemáticas de 
altas dimensiones para interpolar entre las energías de referencia.

**Zuo2020**: Los potenciales interatómicos de ML dan una relación cuantitativa
entre el descriptor del entorno local y la PES. Hay un trade-off entre los grados
de libertad del modelo a usar y el costo computacional que implica.

**Behler2017**: Los métodos de primeros principios están limitados a sistemas
pequeños y su aceleración provino de la evolución en capacidad de computo. Los
métodos de aprendizaje automático para describir las interacciones entre los 
átomos, entrenados con datos de la estructura electrónica, pueden acelerar las
simulaciones en ordenes de magnitud preservando la precisión.

**Chen2020**: Las técnicas de ML se están utilizando para ayudar a encontrar 
correlaciones entre materiales, entender la química y la física de estos y 
acelerar su descubrimiento.

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
    de harmónicos esféricos. Matriz del biespectro.
    - Superposición suave de las posiciones atómicas: Gaussianas en vez de
    funciones delta como en el caso anterior. Hay que incorporar la 
    invariancia rotacional "a mano".
    - Matriz de Coulomb: Autovalores de la matriz (relacionada a la distancia).

**Mishin2021**: sección 3.2
+ Guardan información del entorno local en una cantidad fija de parámetros.
+ Algunos descriptores comunes:
    - Descriptores Gaussianos: combinaciones de funciones gaussianas de 2 y 3
    cuerpos multiplicadas por un radio de corte suave.
    - Descriptores de Zernike: el entorno atómico es descripto mediante funciones
    de Zernike, son computacionalmente más rápidos que el método del biespectro
    y más fáciles de derivar.
    - Tensor momento: se obtienen multiplicando funciones radiales por productos
    externos de los vectores posición de los átomos vecinos.
    - Superposición suave de las posiciones atómicas (SOAP): picos gaussianos de
    densidad superpuestos, que son expandidos en harmónicos esféricos. Son los
    más lentos computacionalmente.
    - Análisis espectral de vecinos (SNAP): similar a SOAP pero con una expansión
    en la base 4D de harmónicos.
    - Expansión de clusters atómicos (ACE): El entorno atómico se representa por 
    polinomios de funciones que forman una base completa, que son producto de una
    función radial y una angular.

**Chen2020**:
+ Los descriptores de las estructuras deben ser invariantes ante traslaciones, 
rotaciones y permutaciones de átomos del mismo tipo.
+ Una opción es construirlos a partir del entorno local de cada átomo:
    - ACSF,
    - coeficientes del biespectro,
    - SOAP,
    - tensor momento,
    - CFID, classical force-field-inspired descriptors,
Estos se benefician de la localidad de las propiedades que se desean reproducir,
por ejemplo, la energía total puede ser dividida en la energía local de cada átomo.

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

**Hong2019**:

- Categorización de ML.
- Instancia de entrenamiento, hipótesis, hiperparametros, función costo, 
feauture y target (característica y objetivo?).
- Aprendizaje automático y Redes neuronals supervisado:
    * pares de ejemplos input-output, 
    * ejemplos de algoritmos,
    * DNN definición (fig 2 para explicar y dar el ejemplo de potenciales).
- No supervisado:
    * clustering y visualización,
- ML flujo de trabajo en ciencias de los materiales:
    * Obtener datos de entrenamiento (DFT),
    * flujo de trabajo automatizado: script de pre y post procesamiento de 
    los datos de, por ej, QE y LAMMPS,
    * error cuadrático medio. 
+ _Algoritmos genéticos_: para encontrar parametros óptimos a potenciales 
conocidos (fig 3 también podría explicar _Symbolic regression_, son parecidos,
aunque acá se usan potenciales conocidos).

**Mishin2021**:
+ _Redes neuronales_: Nodos organizados en capas. feed-forward. El vector con las
propiedades (descriptor) entra en la input layer y la energía del sistema sale
por la output layer. Las capas ocultas en el medio proveen parámetros ajustables.
+ _Physical informed ML potential_: Como _Algoritmos genéticos_ de Hong2019 y 
_Symbolic regression_ de Mueller2020. Esto puede realizarse con cualquier 
potencial ya conocido y con cualquier método de regresión.

**Zuo2020**:
+ La energía potencial se expresa como una suma de energías atómicas que dependen
del entorno local de cada átomo.
+ Los potenciales (en odren cronológico):
    - _Redes neuronales_ (NNP): Como descriptor usan ACSF (atom-centered symmetry
    functions), lo mismo que Behler2016 pero resumido.
    - _GAP_: Usa SOAP como descriptor.
    - _SNAP_: Usa los coeficientes del biespectro de las funciones de densidad
    de los átomos vecinos.
    - _MTP_: tensores rotacionalmente covariantes para describir.
+ Figura 1: esquema para generar los datos de DFT y desarrollas el potencial ML.

**Behler2017**:
_Redes neuronales_:
+ Una feed-forward NN consiste en un número de neuronas, o nodos, artificiales 
acomodadas en capas. Los nodos de la capa de entrada se corresponden con las 
componentes del vector descriptor. La energía potencial se encuntra en la capa 
de salida, que es función del descriptor, donde la forma funcional está dada por
el número de capas ocultas, que no tienen sentido físico, y del número de nodos
en cada una de ellas entre la capa de entrada y la de salida.
+ Para obtener el valor en un nodo se realiza una combinación lineal de los 
nodos de la capa anterior, donde se agrega un bias para acomodar el offset. Luego
se le aplica una función de activación, que son funciones no-lineales continuas
que tienen la propiedad de converger a valores constantes para argumentos muy
positivos y muy negativos, mientras que en el medio tienen una región no-lineal
que permite representar la forma arbitraria de la PES, en este caso.
+ La NN tiene una forma funcional jerárquica anidada de funciones de activación 
que actúan sobre combinaciones lineales.
+ La energía del sistema, además de depender de los elementos del vector 
descriptor, depende de los valores numéricos de los parametros de peso y de bias,
que son determinados mediante una optimización iterativa del gradiente, 
usualmente back-propagation, usando un conjunto de datos conocidos de la energía.
Usando como función objetivo el RMSE de estas energías del conjunto de 
entrenamiento.
+ Para afrontarse las limitaciones de las primeras NN:
    - el número de coordenadas está relacionado al número de grados de libertad,
    - la energía debe ser invariante ante traslaciones y rotaciones, ademas de
    simetría de permutación,
    - sólo pueden ser usadas para un tamaño fijo de problema.
se utilizó una NN separada para cada átomo en el sistema. Cada una de estas redes
neuronales atómicas provee de una contribución a la energía total en función del
entorno local.
+ Distintos descriptores de ese entorno local ya discutidos en **Behler2016**
(un poco se repite).
+ Las fuerzas se calculan usando analiticamente usando la derivada negativa de 
la energía potencial, usando la regla de la cadena se deriva con respecto al 
descriptor y luego el descriptor con respecto a la coordenada. Donde la derivada 
de la energía con respecto al descriptor se obtiene de la estructura de la red 
neuronal y la derivada del descriptor con respecto a la coordenada de su forma
funcional.
+ No es necesario usar estas energías locales para entrenar el modelo.
+ Las interacciones a largo alcance pueden agregarse de manera separada, por
ejemplo la electrostatica, usando una segunda red neuronal atómica entrenada para
este proposito.

+ Challanges:
    - La estructura de un potencial ML consiste escencialmente en dos partes:
        * descriptor estructural,
        * modelo de machine learning
    y hay muchas posibilidades para ambas componentes.
    - La transferibilidad es limitada, la incerteza de la predicción crece fuera 
    del rango en el cual se entrenó.
    - Se puede iterar el conjunto de entrenamiento, haciendo que el mismo crezca
    al usar estructuras en los cuales dos NN distintas divergen entre ellas.

## Conclusiones

**Deringer2020**: Leer de nuevo _What is next?_
