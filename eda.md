# Análisis exploratorio

El análisis exploratorio de datos (EDA, por sus siglas en inglés) permite conocer las imágenes y sus anotaciones antes de interpretar el desempeño de los modelos. En WDSD interesa comprobar la integridad de los datos, evitar que escenas muy parecidas aparezcan tanto en entrenamiento como en prueba y entender las dificultades que plantea identificar cada tornillo.

El recorrido tiene dos partes que se complementan:

1. {doc}`notebooks/01_eda_wdsd`: revisa las imágenes, las máscaras y la similitud entre escenas. Construye la partición agrupada que se utiliza en el análisis posterior y en los entrenamientos, y estudia el tamaño de los objetos y el efecto de cambiar la resolución.
2. {doc}`notebooks/02_eda_instancias`: utiliza esa misma partición para estudiar cada tornillo por separado. Examina las anotaciones, la cantidad de tornillos por imagen, su geometría y separación, y la sensibilidad de las métricas a errores en las máscaras.

Primero se establece qué datos se comparan y cómo se separan; después se analiza qué debe distinguir el modelo dentro de cada imagen.
