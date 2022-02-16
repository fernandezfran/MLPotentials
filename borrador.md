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

**Hong2019**:
+ ML (inductivo) vs programación tradicional (deductivo).
+ Cambios en las definiciones de AI, ML y DL (fig 1).
+ Sistemas atómicos.
+ Sección 2:
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

**Deringer2020**:
+ Para complementar la gran cantidad de herramientas experimentales que hay para
estudiar materiales relevantes para las distintas partes de las baterias se pueden
realizar distintas simulaciones computacionales: DFT o FF (potenciales ajustados
empíricamente).
+ Un modelado emergente y complementario que se presenta son los potenciales
interatómicos de ML, rápidos y precisos creados a partir de datos de referencia
provenientes de cálculos de mecánica cuántica.
+ Potenciales de ML:
    - Aprendizaje supervisado, una tarea de regressión, dados los datos precisos
    de energías y fuerzas para distintos puntos de la PES, se realiza el mejor
    ajuste posible sin asumir formas funcionales específicas de la misma.
    - Las propiedades más simples de la PES, como la repulsión a corta distancia,
    deben ser aprendidas por el modelo.
    - Figura 1: a) esquema del flujo de trabajo, b) distintos métodos.
    - Son computacionalmente más costosos que los potenciales empíricos.

**Mishin2021**:
+ Muchas formas funcionales se han propuesto para potenciales interatómicos 
clásicos para mejorar la precisión.
+ Potenciales interatómicos tradicionales:
    - Figura 1: diagrama para explicar como funciona un potencial.
    - Breve explicación de para que se puede usar cada potencial clásico (los 
    nombrados también por Mueller2020).
    - Cómo son los parametros de estos potenciales.
    - Suelen ser transferibles.

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

## Aplicaciones

**Deringer2020**: Aplicaciones a baterias [Refs 47-52]
+ Difusión de Li en Li3PO4 y RDF (ML NN-type) [53].
+ Difusión de Li en C desordenado (ML GAP) [57]
+ Difusión de Li en distintas estructuras con LOTF (Learn on the fly) ML model [54]
+ a-LiSi (ANN) curvas de voltaje [47]
Si son sólo este tipo de aplicaciones las que muestro, el título del seminario
puede ser: **Potencials interatómicos de aprendizaje automatizado y su aplicación
al estudio de baterias de Li**.

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

**Hong2019**:
El desarrollo tradicional de potenciales empíricos puede beneficiarse de los 
algoritmos geneticos de ML o se pueden obtener directamente potenciales de ML
ajustando directamente la PES obtenida de la estructura electrónica.

**Deringer2020**: Leer de nuevo _What is next?_

**Mishin2021**:
Las predicciones de los potenciales de ML fuera del rango en el que fueron 
entrenados pueden llevar a resultados que no son físicamente razonables. Los 
potenciales de ML informados con física pueden mejorar la transferibilidad.
