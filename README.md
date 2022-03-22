# Potenciales interatómicos de aprendizaje automático y su aplicación a baterias de litio

> Seminario del doctorado en Física en [FAMAF](https://www.famaf.unc.edu.ar/).

En el campo de las simulaciones computacionales existen principalmente dos 
variantes para el estudio de materiales, por primeros principios o a través de 
aproximaciones clásicas, las primeras de ellas de gran precisión pero limitada a 
sistemas pequeños mientras que la segunda permite simulaciones en escalas más 
grandes pero su precisión depende de la forma funcional que se elija. Debido a la
complejidad en aumento de los sistemas electroquímicos de interés en el área de 
las baterias de litio, es necesario que las simulaciones puedan realizarse a 
escalas mayores sin perder precisión, los potenciales interatómicos de aprendizaje 
automático ofrecen representar la superficie energía-potencial mediante un
entrenamiento con datos de la estructura electrónica que permite llevar esto a 
cabo. En este seminario se introducen dichos potenciales y se presentan distintas
aplicaciones de los mismos en distintas componentes de las baterias de litio. 

----------------------------------------------------------------------------------
### Referencias

- Deringer, V. L. (2020). Modelling and understanding battery materials with 
machine-learning-driven atomistic simulations. _Journal of Physics: Energy_, 2(4), 
041003.

- Deringer, V. L., Caro, M. A., \& Csányi, G. (2019). Machine learning interatomic 
potentials as emerging tools for materials science. _Advanced Materials_, 31(46), 
1902765.

- Mishin, Y. (2021). Machine-learning interatomic potentials for materials 
science. _Acta Materialia_, 214, 116980.

- Behler, J. (2017). First principles neural network potentials for reactive 
simulations of large molecular and condensed systems. _Angewandte Chemie 
International Edition_, 56(42), 12828-12840.

- Behler, J. (2016). Perspective: Machine learning potentials for atomistic 
simulations. _The Journal of chemical physics_, 145(17), 170901.

- Mueller, T., Hernandez, A., \& Wang, C. (2020). Machine learning for 
interatomic potential models. _The Journal of chemical physics_, 152(5), 050902.

- Hong, Y., Hou, B., Jiang, H., \& Zhang, J. (2020). Machine learning and 
artificial neural network accelerated computational discoveries in materials 
science. _Wiley Interdisciplinary Reviews: Computational Molecular Science_, 
10(3), e1450.

- Botu, V., Batra, R., Chapman, J., \& Ramprasad, R. (2017). Machine learning 
force fields: construction, validation, and outlook. _The Journal of Physical 
Chemistry C_, 121(1), 511-522.

- Li, W., Ando, Y., Minamitani, E., \& Watanabe, S. (2017). Study of Li atom 
diffusion in amorphous Li3PO4 with neural network potential. _The Journal of 
chemical physics_, 147(21), 214106.

- Fujikake, S., Deringer, V. L., Lee, T. H., Krynski, M., Elliott, S. R., \& 
Csányi, G. (2018). Gaussian approximation potential modeling of lithium 
intercalation in carbon nanostructures. _The Journal of chemical physics_, 
148(24), 241714.

- Artrith, N., Urban, A., \& Ceder, G. (2018). Constructing first-principles 
phase diagrams of amorphous Li x Si using machine-learning-assisted sampling with
an evolutionary algorithm. _The Journal of chemical physics_, 148(24), 241711.


----------------------------------------------------------------------------------
### Compilación

Para compilar se puede utilizar el **Makefile** simplemente tipeando en la 
terminal (Linux OS):
```bash
make
```
para borrar los archivos generados utilizar `make clean`.
