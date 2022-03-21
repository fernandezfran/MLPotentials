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

## Potenciales de ML

**Behler2016**:
+ _Potenciales de aproximación gaussiana y métodos kernel_: Combina algún
descriptor con un kernel que relaciona la estructura con la energía. Es un 
suma pesada sobre las energías conocidas de los entornos del conjunto de 
entrenamiento. Combinación lineal de funciones que dependen del descriptor.
