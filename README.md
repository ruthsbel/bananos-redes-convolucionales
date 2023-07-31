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

## Análisis del clasificador
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
Luego, se definen las transformaciones que se aplicarán a las imágenes de entrenamiento y prueba. Para el conjunto de entrenamiento, se realiza una redimensión de las imágenes a un tamaño de 120x120 píxeles; se aplica RandomHorizontalFlip para aumentar el conjunto de datos y mejorar la generalización del modelo; luego, se convierten las imágenes en tensores de PyTorch y se normalizan los valores de los píxeles para que estén en el rango de [-1, 1]. Para el conjunto de prueba, solo se aplica la redimensión y la normalización.

```python
train_transforms = transforms.Compose([transforms.Resize(size = (120,120)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
                                       ])

test_transforms = transforms.Compose([ transforms.Resize(size = (120,120)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
                                       ])
```
Ahora, se divide el conjunto de entrenamiento en un subconjunto de entrenamiento y validación. Se utiliza SubsetRandomSampler para seleccionar aleatoriamente los índices de las imágenes que formarán el subconjunto de entrenamiento y el subconjunto de validación.

```python
valid_size = 0.2
train_length = len(train_data)
indices = [i for i in range(train_length)]
np.random.shuffle(indices)
split = int(np.floor(valid_size * train_length))
train_idx = indices[split:]
valid_idx = indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
```

En este paso, se utiliza DataLoaderpara cargar los datos en lotes. Se crean 3 DataLoader: uno para el conjunto de entrenamiento, otro para el conjunto de validación y otro para el conjunto de prueba. Específicamente, trainloader y validloader utilizan el subconjunto de entrenamiento y validación respectivamente, mientras que testloader carga el conjunto de prueba. También se especifica el tamaño del lote (batch_size) para cada DataLoader.

```python
trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, sampler = train_sampler)
validloader = torch.utils.data.DataLoader(train_data, batch_size = 64, sampler = valid_sampler)
testloader = torch.utils.data.DataLoader(test_data, batch_size = 100, shuffle = True)
```

## Diseñando la arquitectura de la Red Neuronal Convolucional
### Modelo 1

Para el primer modelo, se define la arquitectura de la red convolucional, aplicando 3 capas convolucionales y una capa de Max-Pooling para reducir la dimensionalidad. Posteriormente, se definen 2 capas lineales completamente conectadas para el flattening y dropout para evitar un sobreajuste del modelo. Luego, se utiliza ReLU para que el modelo pueda desarrollar mejor y aprenda mucho más rápido.

```python
#(64 x 3 x 120x120) -> conv1/maxPool -> (64 x 16 x 60x60) -> conv2/maxPool -> (64 x 32 x 30x30) -> conv3/maxPool -> (64 x 64 x 15x15) -> Flat -> (64x15x15)-> fc1 -> (100)->fc2 -> (2)
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, 3, padding = 1)
    self.conv2 = nn.Conv2d(16, 32, 3,padding = 1)
    self.conv3 = nn.Conv2d(32, 64, 3,padding = 1)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(64 * 15 * 15, 100)
    self.fc2 = nn.Linear(100, 2)
    self.dropout = nn.Dropout(0.25)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))
    x = self.dropout(x)
    x = x.view(-1, 64 * 15 * 15)
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    return x
```
```python
Resultados para el modelo 1
- Accuracy de prueba para Exportacion: 66% (33/50)
- Accuracy de prueba para Rechazo: 90% (45/50)
- Accuracy total para la prueba: 78% (78/100)
```

### Modelo 2
Para el segundo modelo, se definen 5 capas convolucionales y una capa de Max-Pooling para reducir la dimensionalidad. Posteriormente, se definen 4 capas lineales completamente conectadas para el flattening y dropout para evitar un sobreajuste del modelo. Luego, se utiliza igualmente ReLU para que el modelo pueda desarrollar mejor y aprenda mucho más rápido.

```python
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(3,16,3,padding=1)
    self.conv2 = nn.Conv2d(16,32,3,padding=1)
    self.conv3 = nn.Conv2d(32,64,3,padding=1)
    self.conv4 = nn.Conv2d(64,128,3,padding=1)
    self.conv5 = nn.Conv2d(128,256,3,padding=1)
    self.pool = nn.MaxPool2d(2,2)
    self.fc1 = nn.Linear(256*15*15, 500)
    self.fc2 = nn.Linear(500,256)
    self.fc3 = nn.Linear(256,100)
    self.fc4 = nn.Linear(100,2)
    self.dropout = nn.Dropout(0.20)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))
    x = F.relu(self.conv4(x))
    x = F.relu(self.conv5(x))
    x = self.dropout(x)
    x = x.view(-1,256*15*15)
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = F.relu(self.fc2(x))
    x = self.dropout(x)
    x = F.relu(self.fc3(x))
    x = self.dropout(x)
    x = self.fc2(x)
    return x
```
```python
Resultados para el modelo 2
Epoch: 1 	Training Loss: 3.957485 	Validation Loss: 0.705334
Error de validación fue mejor, guardando modelo inf -> 0.705334
Epoch: 2 	Training Loss: 0.699826 	Validation Loss: 0.693607
Error de validación fue mejor, guardando modelo 0.705334 -> 0.693607
Epoch: 3 	Training Loss: 0.697879 	Validation Loss: 0.694038
Epoch: 4 	Training Loss: 0.694033 	Validation Loss: 0.693077
Error de validación fue mejor, guardando modelo 0.693607 -> 0.693077
Epoch: 5 	Training Loss: 0.703862 	Validation Loss: 0.693551
Epoch: 6 	Training Loss: 0.692807 	Validation Loss: 0.696675
Epoch: 7 	Training Loss: 1.073586 	Validation Loss: 0.693379
Epoch: 8 	Training Loss: 0.693268 	Validation Loss: 0.693084
Epoch: 9 	Training Loss: 0.693315 	Validation Loss: 0.693165
Epoch: 10 	Training Loss: 0.693215 	Validation Loss: 0.693238
Epoch: 11 	Training Loss: 0.692939 	Validation Loss: 0.693331
Epoch: 12 	Training Loss: 0.694360 	Validation Loss: 0.693354
Epoch: 13 	Training Loss: 0.694432 	Validation Loss: 0.693138

- Accuracy de prueba para Exportacion:  2% ( 1/50)
- Accuracy de prueba para Rechazo: 78% (39/50)
- Accuracy total para la prueba: 40% (40/100)

```

### Modelo 3
Para el segundo modelo, se definen 5 capas convolucionales y una capa de Max-Pooling para reducir la dimensionalidad. Posteriormente, se definen 2 capas lineales completamente conectadas para el flattening y dropout para evitar un sobreajuste del modelo. Luego, se utiliza igualmente ReLU para que el modelo pueda desarrollar mejor y aprenda mucho más rápido.

```python
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(3,16,3,padding=1)
    self.conv2 = nn.Conv2d(16,32,3,padding=1)
    self.conv3 = nn.Conv2d(32,64,3,padding=1)
    self.conv4 = nn.Conv2d(64,128,3,padding=1)
    self.conv5 = nn.Conv2d(128,256,3,padding=1)
    self.pool = nn.MaxPool2d(2,2)
    self.fc1 = nn.Linear(256*15*15, 100)
    self.fc2 = nn.Linear(100,2)
    self.dropout = nn.Dropout(0.20)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))
    x = F.relu(self.conv4(x))
    x = F.relu(self.conv5(x))
    x = self.dropout(x)
    x = x.view(-1,256*15*15)
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    return x
```
```python
Resultados para el modelo 3
Epoch: 1 	Training Loss: 3.957485 	Validation Loss: 0.705334
Error de validación fue mejor, guardando modelo inf -> 0.705334
Epoch: 2 	Training Loss: 0.699826 	Validation Loss: 0.693607
Error de validación fue mejor, guardando modelo 0.705334 -> 0.693607
Epoch: 3 	Training Loss: 0.697879 	Validation Loss: 0.694038
Epoch: 4 	Training Loss: 0.694033 	Validation Loss: 0.693077
Error de validación fue mejor, guardando modelo 0.693607 -> 0.693077
Epoch: 5 	Training Loss: 0.703862 	Validation Loss: 0.693551
Epoch: 6 	Training Loss: 0.692807 	Validation Loss: 0.696675
Epoch: 7 	Training Loss: 1.073586 	Validation Loss: 0.693379
Epoch: 8 	Training Loss: 0.693268 	Validation Loss: 0.693084
Epoch: 9 	Training Loss: 0.693315 	Validation Loss: 0.693165
Epoch: 10 	Training Loss: 0.693215 	Validation Loss: 0.693238
Epoch: 11 	Training Loss: 0.692939 	Validation Loss: 0.693331
Epoch: 12 	Training Loss: 0.694360 	Validation Loss: 0.693354
Epoch: 13 	Training Loss: 0.694432 	Validation Loss: 0.693138
Epoch: 14 	Training Loss: 0.693146 	Validation Loss: 0.693100
Epoch: 15 	Training Loss: 0.693661 	Validation Loss: 0.693099

- Accuracy de prueba para Exportacion: 64% (32/50)
- Accuracy de prueba para Rechazo: 50% (25/50)
- Accuracy total para la prueba: 57% (57/100)

```

Se crea una instancia del modelo Net y muestra una descripción de la arquitectura de la red utilizando print(model). Luego, se utiliza .cuda() para cargar el modelo en la GPU para entrenar la red de manera más rápida y eficiente.

```python
model = Net()
print(model)
model.cuda()
```

Se guarda el modelo entrenado:

```python
torch.save(model, "/content/model.pt")
```

## Definiendo los hiperparámetros del modelo

El proceso para establecer los hiperparámetros para entrenar la red convolucional utilizando PyTorch es el siguiente:

Primero se importa las bibliotecas necesarias de PyTorch para trabajar con optimizadores y definir el criterio de pérdida (loss).

En este caso, se utiliza nn.CrossEntropyLoss() como criterio de pérdida durante el entrenamiento del modelo, que es adecuado para problemas de clasificación multiclase y nos proporcionará probabilidades al final de la red.

```python
criterion = nn.CrossEntropyLoss()
```

Se utiliza el optimizador Adam con un learning rate de 0.001. El learning rate controla el tamaño de los pasos que el optimizador toma al ajustar los pesos y es un hiperparámetro crítico que afecta el rendimiento del modelo. Se realizaron pruebas con otros learning rates, pero con 0.001 se produjeron los mejores resultados en este caso particular.

```python
optimizer = optim.Adam(model.parameters(), lr = 0.001)
```

Se definen 15 épocas (n_epochs = 15), siendo el mejor valor para el modelo.

## Entrenando y validando el modelo

Se inicializa la variable valid_loss_min con un valor infinito (np.inf). Esta variable se utilizará para realizar un seguimiento del valor mínimo de la pérdida de validación durante el entrenamiento del modelo.

```python
valid_loss_min = np.inf
```

Se inicia el proceso de entrenamiento y validación del modelo. Se realiza un bucle a través de los epochs (épocas) y, dentro de cada epoch, se itera sobre los datos de entrenamiento y de validación. Durante el entrenamiento, se calcula la pérdida y se actualizan los pesos del modelo utilizando el optimizador. Durante la validación, se calcula la pérdida en el conjunto de validación para evaluar el rendimiento del modelo en datos no vistos.

```python
for epoch in range(1, n_epochs + 1):
    train_loss = 0.0
    valid_loss = 0.0
    model.train()

    # Bucle de entrenamiento
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    model.eval()

    # Bucle de validación
    for batch_idx, (data, target) in enumerate(validloader):
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        valid_loss += loss.item() * data.size(0)

    # Cálculo de la pérdida promedio
    train_loss = train_loss / len(trainloader.sampler)
    valid_loss = valid_loss / len(validloader.sampler)

    # Impresión de resultados
    print("Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(epoch, train_loss, valid_loss))

    # Guardar el mejor modelo de validación
    if valid_loss <= valid_loss_min:
        print("Error de validación fue mejor, guardando modelo {:.6f} -> {:.6f}".format(valid_loss_min, valid_loss))
        torch.save(model.state_dict(), "mejor_modelo.pt")
        valid_loss_min = valid_loss
```

## ¿El modelo está subajustado (underfitting), sobreajustado (overfitting) o es adecuado (right)?

Como resultado se pudo detectar que, con más épocas (alrededor de 15), el modelo sigue aprendiendo frente al conjunto de datos de entrenamiento. En relación al conjunto de datos de validación, se pudo detectar que el modelo empeora implicando un sobreajuste (overfitting), lo cual justifica que se guarde el mejor modelo de validación para luego aplicarlo.

Esto puede deberse a que no se tomaron demasiados datos para el entrenamiento del modelo. Utilizar un dataset más limpio y con más validaciones, los datos del modelo mejorarán.

Se realizaron varias pruebas cambiando el learning rate y el número de epochs; finalmente se hizo el entrenamiento con 30 epochs para demostrar que pasados los 15 epochs, el modelo no mejora.

## Evaluando el modelo

Para evaluar el rendimiento del modelo, se carga el modelo previamente entrenado y se inicializan las variables test_loss, class_correct y class_total con valores iniciales. test_loss se utilizará para calcular la pérdida promedio en el conjunto de prueba, mientras que class_correct y class_total se utilizarán para calcular la precisión (accuracy) de cada clase en el conjunto de prueba.

```python
model.load_state_dict(torch.load("modelo3.pt"))
test_loss = 0.0
class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))
```

Luego, se pone el modelo en modo de evaluación (model.eval()) para desactivar el ajuste de los pesos del modelo. Se itera a través del conjunto de prueba (testloader) y se obtiene la salida del modelo para cada lote de datos. Posteriormente, se calcula la pérdida para cada lote y se acumula en test_loss. También se calcula la precisión de predicción para cada clase y se acumula en class_correct y class_total.

```python
model.eval()

for data, target in testloader:
    data, target = data.cuda(), target.cuda()
    output = model(data)
    loss = criterion(output, target)
    test_loss += loss.item() * data.size(0)
    _, pred = torch.max(output, 1)
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())

    for i in range(100):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1
```

Ahora, se calcula la pérdida promedio en el conjunto de prueba dividiendo test_loss entre el tamaño total del conjunto de prueba (len(testloader.dataset)). Luego, se imprime la pérdida promedio.

También se calcula la precisión (accuracy) para cada clase en el conjunto de prueba y se imprime en forma de porcentaje. Si no hay muestras para una clase específica, se imprime “N/A (no hubo muestras)”.

Finalmente, se calcula y se imprime la precisión total del modelo en el conjunto de prueba.

```python
test_loss = test_loss / len(testloader.dataset)
print("Perdida de test: {:.6f}\n".format(test_loss))

for i in range(2):
    if class_total[i] > 0:
        print("Accuracy de prueba para %5s: %2d%% (%2d/%2d)" % (classes[i], 100 * class_correct[i] / class_total[i],
                                                               np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print("Accuracy de prueba para %5s: N/A (no hubo muestras)" % classes[i])

print("\nAccuracy total para la prueba: %2d%% (%2d/%2d)" % (
    100. * np.sum(class_correct) / np.sum(class_total), np.sum(class_correct), np.sum(class_total)))
```

### Resultados del Modelo 1

En este caso, los resultados obtenidos indican que la precisión (accuracy) de prueba para la clase Exportación es del 66%, lo que significa que el modelo acertó correctamente 33 muestras de un total de 50 imágenes de prueba. En cambio, la precisión (accuracy) de prueba para la clase Rechazos es del 90% , lo que indica que el modelo acertó correctamente 45 muestras de un total de 50. Finalmente, la precisión total para el conjunto de prueba es del 78%, acertando 78 muestras de un total de 100.

### Resultados del Modelo 2
En este caso, los resultados obtenidos indican que la precisión (accuracy) de prueba para la clase Exportación es del 2%, lo que significa que el modelo acertó correctamente 1 muestra de un total de 50 imágenes de prueba. En cambio, la precisión (accuracy) de prueba para la clase Rechazos es del 78% , lo que indica que el modelo acertó correctamente 39 muestras de un total de 50. Finalmente, la precisión total para el conjunto de prueba es del 40%, acertando 40 muestras de un total de 100.

## Visualización del grafo de una Red Neuronal con Torchviz
La visualización del grafo permite comprender la arquitectura de la red y la secuencia de operaciones que ocurren durante el flujo hacia adelante (forward pass).

Luego, debe asegurarse de que la biblioteca Torchviz esté instalada en su entorno de trabajo e importar las bibliotecas necesarias para visualizar el grafo. torchview proporciona la función draw_graph() que permitirá crear el grafo visual de la Red Neuronal. También se importa la biblioteca graphviz para establecer el formato de salida como imagen PNG para la visualización en el entorno Jupyter.

```python
!pip install torchviz
import torchvision
from torchview import draw_graph
import graphviz
graphviz.set_jupyter_format('png')
```

Se crea el grafo de la red utilizando la función draw_graph(). Se proporciona una instancia del modelo Net() que representa la arquitectura de la red neuronal que se quiere visualizar. También se especifica el tamaño de entrada de la red en input_size, que en este caso es (64, 3, 120, 120). Los argumentos hide_inner_tensors y hide_module_functions permiten controlar la visualización de los tensores internos y las funciones de los módulos de la red, respectivamente.

```python
model_graph_1 = draw_graph(
    Net(), input_size=(64, 3, 120, 120),
    graph_name='MLP',
    hide_inner_tensors=False,
    hide_module_functions=False,
)
```

Ahora, se procede a generar el grafo en formato de imagen del primer modelo:

```python
model_graph_1.visual_graph
```

### Grafo del Modelo 1
![grafo](https://github.com/ruthsbel/bananos-redes-convolucionales/assets/10469932/8ef31eb0-b90d-4aa8-8eaa-9ffacd07f49d)

### Grafo del Modelo 2
![grafo](https://github.com/ruthsbel/bananos-redes-convolucionales/assets/10469932/7d1c2d3e-67fb-4219-869d-84ce0ed04c4e)

### Grafo del Modelo 3
![modelo3](https://github.com/ruthsbel/bananos-redes-convolucionales/assets/10469932/dee9af12-6ba1-4b79-a154-733719256133)


## Conclusiones
El primer modelo se ha seleccionado como el mejor luego de haber efectuado las pruebas debido a que muestra un mejor rendimiento hasta la época 14–15, luego de eso tiende al sobreajuste. Además, el accuracy para detectar banano que debe ser rechazado es bastante alto (90%) siendo precisamente lo que se busca para el clasificador ya que, se pretende que se exporte el banano de la variedad Cavendish de la más alta calidad; por lo tanto, si el clasificador es bastante estricto al momento de seleccionar los productos que serán rechazados se garantiza que a los mercados internacionales llegue el mejor producto desde los puertos ecuatorianos.

### Gráfico de entrenamiento y validación
![grafico](https://github.com/ruthsbel/bananos-redes-convolucionales/assets/10469932/747aef7e-bfe9-415a-8391-36df3f149767)

## ¿Cuál es el mejor modelo?
La siguiente imagen muestra un resumen de los resultados obtenidos con los tres modelos que se entrenaron:

+ El enfoque del modelo de clasificación es identificar la mayor cantidad de bananos de rechazo, ya que eso influye más en el modelo de negocio de las bananeras según lo planteado en el proyecto.
+ Con esta premisa, se selecciona el modelo que tenga el mejor accuracy para identificar el banano de rechazo.
+ El modelo A que representa una red de 3 capas convolucionales ofrece una mejor accuracy que el resto de modelos.

![grafico](https://github.com/ruthsbel/bananos-redes-convolucionales/assets/10469932/b7b46da7-fbcc-43dd-85e3-dd4b544955e4)
Comparación de los valores de accuracy de los modelos entrenados para clasificar el banano de la variedad Cavendish. Se visualiza que el primer modelo tiene un mejor accuracy para clasificar banano apto para la exportación y el banano que deberá ser rechazado. Fuente. Elaboración propia.

### Referencias bibliográficas
+ [1] León Serrano, L.A., Arcaya Sisalima, M.F., Barbotó Velásquez, N.A., Bermeo Pineda, Y.L. Ecuador: Análisis Comparativo de Las Exportaciones de Banano Orgánico y Convencional e Incidencia En La Balanza Comercial, 2018 (2020) Rev. Científica Tecnológica UPSE, 7, pp. 38–46.

+ [2] K. Veliz, L. Chico-Santamarta, A. D. Ramirez, “The Environmental Profile of Ecuadorian Export Banana: A Life Cycle Assessment,” Foods, vol. 11, no. 20, p. 3288, Oct. 2022, DOI: 10.3390/foods11203288.
