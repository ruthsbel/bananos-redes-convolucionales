# Clasificando bananos de la variedad cavendish usando modelos de redes neuronales convolucionales que categoricen basado en patrones
## Un poco de contexto
Ecuador es el principal exportador de banano del mundo según datos de la FAO, lo que representa para el país uno de los principales ingresos y asegurarse de que su calidad sea la mejor es parte vital del proceso de la producción del producto.
La clasificación del banano de exportación actualmente es manual, se determina visualmente si un banano es o no de exportación, esta solución plantea un modelo que pueda determinar mediante fotos si un banano se puede exportar o no.
## Planteamiento inicial
Planteamos clasificar si un banano de la variedad cavendish puede ser exportado o no utilizando modelos de redes neuronales convolucionales que tomen decisiones basados en patrones que se identifican en imágenes previamente tomadas a bananos que se han clasificado como bananos de exportación o no.

Este artículo tiene como objetivo explorar cómo las redes neuronales convolucionales están transformando la clasificación de los bananos Cavendish, abriendo nuevas oportunidades para lograr resultados más rápidos y precisos. Para comprender plenamente la importancia de esta revolución en la industria bananera de Ecuador, examinaremos los avances en técnicas computacionales, especialmente en el aprendizaje automático y profundo, que han allanado el camino para el uso de redes neuronales convolucionales en la categorización precisa de los bananos Cavendish.

## Motivación del proyecto
En la actualidad, el proceso de clasificación de banano en Ecuador no está automatizado; es por ello que la categorización precisa del banano de tipo Cavendish garantiza que a los mercados internacionales llegue un producto de alta calidad. Sin embargo, el proceso tradicional de clasificación realizado por especialistas requiere de una gran cantidad de tiempo, esfuerzo y recursos, convirtiéndose en un desafío la satisfacción de la creciente demanda del producto y mantener los altos estándares de calidad. La utilización de la CNN para la categorización del banano de tipo Cavendish podría representar una revolución para la industria bananera, permitiendo una clasificación más rápida y precisa.
## Descripción del problema
Ante la importancia de la exportación de banano en Ecuador y la poca agilidad en el proceso de clasificación del banano de la variedad Cavendish, se han podido detectar los siguientes problemas que se pretenden resolver con la propuesta:

+ La clasificación manual de los bananos para determinar si son aptos para la exportación o no, es un proceso que está propenso a errores humanos.
+ Las personas que están a cargo de la clasificación del banano pueden tener diversos tipos de criterios al momento de evaluar si el banano califica o es rechazado. Esto puede conducir a una clasificación inconsistente, afectando la calidad del producto y su reputación a nivel internacional.
+ La clasificación manual puede ser un proceso lento y laborioso causando retrasos en la cadena de suministros y afectar los tiempos de entrega del producto a mercados internacionales.
+ Teniendo en cuenta que Ecuador compite con países que pueden tener mayores recursos económicos y tecnológicos, el sistema de clasificación manual podría disminuir el nivel de competitividad con otros países exportadores de banano perdiendo a muchos compradores y teniendo que recurrir a otros métodos como bajar los precios de banano para compensar esta desventaja.

## Objetivos del proyecto
Como equipo se ha decidido que el proyecto deberá cumplir con los siguientes objetivos:

+ Entrenar un clasificador de imágenes para identificar el banano de la variedad Cavendish que es apto para la exportación e identificar aquellos bananos que no cumplen con los estándares marcados por el mercado internacional.
+ Disminuir los tiempos de clasificación del banano, agilizando y optimizando este proceso, liberando recursos y tiempo valioso para los involucrados en la cadena de suministro del banano.
+ Ahorro en los costos del precio del banano reduciendo la dependencia de la mano de obra para la clasificación, disminuyendo los gastos operativos y permitiendo una mayor eficiencia en la cadena de suministro.
+ Mejorar la calidad del banano que se exporta al detectar, mediante el clasificador, detalles visuales y características específicas que indiquen el estado óptimo de madurez, tamaño, forma y calidad del banano, garantizando que solo los productos de alta calidad sean exportados.

![Dataset a ser utilizado en la red](https://github.com/ruthsbel/bananos-redes-convolucionales/assets/10469932/3762fca9-e730-447a-ab7c-38df3e0accf7)

## Descripción del Dataset
El dataset “Banana Classification Dataset” es una recopilación exhaustiva de imágenes o fotografías que representan bananos de la variedad Cavendish recién cosechados, destinados tanto para su exportación como para su rechazo. Este conjunto de datos ha sido cuidadosamente creado y etiquetado para facilitar la clasificación precisa y diferenciada de los bananos según su estado de aptitud para la exportación. El datadet incluye una amplia variedad de imágenes de bananos, capturadas en distintos momentos después de la cosecha y bajo diversas condiciones de iluminación y fondo.

Las imágenes de bananos aptos para exportación han sido seleccionadas meticulosamente para mostrar frutos en su óptimo estado de madurez y calidad, con una apariencia atractiva y libre de imperfecciones. Estas imágenes representan la categoría de bananos que cumplen con todos los criterios para ser exportados, manteniendo los altos estándares de excelencia que se esperan de los bananos de la variedad Cavendish.

Por otro lado, las imágenes de bananos que deben ser rechazados muestran frutos con diversas anomalías como daños físicos, manchas, irregularidades en el tamaño y la forma, y signos de deterioro. Estas imágenes representan la categoría de bananos que no cumplen con los requisitos para su exportación y, por lo tanto, no son aptos para el mercado internacional.

Los valores del dataset que se utilizaron para este clasificador los puede encontrar en:
[Kaggle](https://www.kaggle.com/datasets/andreruizcalle/banana-classification-dataset?resource=download)

## Estructura del proyecto
Uno de los algoritmos principales que se pueden utilizar para hacer reconocimiento y clasificación de imágenes son las redes neuronales convolucionales (CNN). Las CNN son un tipo de arquitectura de aprendizaje profundo diseñado para efectuar tareas como el reconocimiento de imágenes y la detección de objetos de manera computarizada.
![banano](https://github.com/ruthsbel/bananos-redes-convolucionales/assets/10469932/fbaba164-d815-4b57-84ef-2fa255748523)

#Análisis del clasificador
El clasificador se lo ha desarrollado en Google Colab, para aprovechar el tipo de entorno y las GPU de Google.

Se deben importar las librerías necesarias para el funcionamiento del proyecto. En este caso, se incluyen bibliotecas de PyTorch como torch.utils.data y torchvision, que se utilizarán para cargar y transformar los datos de imágenes. También se importanfastdownload para descargar el conjunto de datos, fastai.vision.all para trabajar con imágenes y modelos, numpy para operaciones matemáticas y fastcore.all para funciones esenciales de Fastai:

```python
from torch.utils.data import SubsetRandomSampler
from torchvision import datasets, transforms
from fastdownload import download_url
from fastai.vision.all import *
import numpy as np
from fastcore.all import *
```
