# Lab2_DL
Lab 2 de Deep Learning

---

## Autores:

- Nelson García 22434
- Joaquín Puente 22296
- 
---

## Respuestas a preguntas

### Parte 1:

- ¿Por qué creen que es el mal rendimiento de este modelo?
  R//
 - Entradas sin normalizar: Se están usando los píxeles en rango [0,255]. Es fácil que las activaciones y los gradientes exploten o se saturen.
 - Arquitectura muy sencilla y pocos datos: Solo un conv+pool y una FC para distinguir gatos/perros, con resolución 16×16. Eso limita mucho la capacidad de extracción de características.  
 - Learning rate y weight‑init poco adecuados: Con learning rate = 0.01 y filtros iniciados en 0.01, parece que los gradientes divergieron (coste sube al segundo epoch y se estanca).

- ¿Qué pueden hacer para mejorarlo?
  R//
- Normalizar inputs: X_train = X_train.astype('float32') / 255.
- Reducir el learning rate (por ejemplo a 1e‑3 o 1e‑4) o usar un optimizador con decaimiento (Adam, RMSProp).
- Aumentar la capacidad: agrega otra capa conv+pool, incrementa filtros (32,64), sube la resolución a 32×32 o 64×64.
- Entrenar más epochs y aplicar técnicas de regularización (dropout, data augmentation).
  
- ¿Cuáles son las razones para que el modelo sea tan lento?
  R//
- Implementación en “crudo” con bucles en Python: Cada conv_forward, conv_backward, pool_forward, pool_backward recorre m×h×w×c con cuatro niveles de loops.
- Sin vectorización ni im2col: Los frameworks (TensorFlow/PyTorch) internamente usan kernels optimizados en C/CUDA.
- Padding excesivo e innecesario: Al hacer pad=2 en un input 16×16 la convolución crece a 20×20 antes de pool. Multiplica el cómputo sin mejorar la representación.

### Parte 2:

- ¿Qué haría para mejorar el rendimiento del modelo?
- ¿Qué haría para disminuir las posibilidades de overfitting?

---

## Dependencias y como correr

Se puede usar Google Colab o usar un entorno virtual.

```bash
pip install virtualenv
```

Con venv

```bash
python -m venv venv
source venv/bin/activate
```

Dentro del entorno virtual:

```bash
pip install jupyterlab
```

Levanta el servidor:

```bash
jupyter lab
```

Instalar librerías:
```bash
pip install -r requirements.txt
```

