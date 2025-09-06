# Anexo B: Manual de Usuario

## 1. Introducción

Este manual proporciona una guía detallada para la instalación, configuración y uso de la plataforma MLOps desarrollada en este proyecto. El objetivo es permitir que investigadores, estudiantes y desarrolladores puedan desplegar y operar el sistema en un entorno local, aprovechando sus capacidades para automatizar el ciclo de vida de modelos de Machine Learning aplicados al análisis de datos de neurociencia.

La plataforma ha sido diseñada para ser portable y autocontenida, minimizando la complejidad de la configuración. A través de los siguientes capítulos, se explicará paso a paso cómo poner en marcha la infraestructura completa, interactuar con la interfaz de usuario para gestionar modelos y ejecutar predicciones.

Es importante destacar que, si bien este proyecto se entrega con un pipeline preconfigurado para el análisis de señales EEG, la arquitectura subyacente es agnóstica al modelo y al dominio. Ha sido diseñada intencionadamente como un "meta-pipeline" genérico y reutilizable. Esto significa que los desarrolladores pueden adaptar la plataforma para orquestar sus propios modelos, utilizando diferentes funciones de carga de datos, preprocesamiento y entrenamiento sin necesidad de modificar la lógica de orquestación principal. Más adelante en este manual se detallará cómo realizar estas personalizaciones.

## 2. Requisitos del Sistema

Antes de proceder con la instalación, es necesario asegurarse de que el sistema anfitrión cumple con los siguientes requisitos mínimos:

*   **Software:**
    *   **Docker Engine:** Versión 20.10.0 o superior.
    *   **Docker Compose:** Versión 1.29.0 o superior (o la versión integrada en Docker Desktop).
    *   **Git:** Para clonar el repositorio del proyecto.

*   **Hardware (Recomendado):**
    *   **Sistema Operativo:** Linux, macOS o Windows con WSL2 (Subsistema de Windows para Linux).
    *   **Memoria RAM:** 8 GB o más, especialmente para manejar el procesamiento de datasets de EEG y la ejecución simultánea de todos los servicios.
    *   **Espacio en Disco:** 40 GB de espacio libre para almacenar las imágenes de Docker, los artefactos de los modelos, los metadatos y los datasets de prueba.

## 3. Instalación y Configuración Inicial

La plataforma utiliza Docker para encapsular cada uno de sus componentes (backend, frontend, base de datos, etc.) en contenedores, y Docker Compose para orquestar su despliegue y comunicación. Este enfoque garantiza un entorno de ejecución consistente y reproducible.

### 3.1. Obtención del Código Fuente

El primer paso es descargar el código fuente del proyecto desde su repositorio oficial. Abra una terminal y ejecute el siguiente comando:

```bash
git clone https://github.com/imEag/MLOps.git
cd MLOps
```

Este comando creará una carpeta llamada `MLOps` en su directorio actual y navegará dentro de ella.

### 3.2. Configuración del Entorno

La plataforma se configura mediante variables de entorno, lo que permite ajustar parámetros como los puertos de red y las credenciales de la base de datos sin modificar el código. El archivo `docker-compose.yml` está preparado para leer estas variables desde un archivo `.env`.

1.  **Crear el archivo de configuración:**
    El proyecto incluye un archivo de ejemplo llamado `.env.example` dentro de la carpeta `backend/`. Cópielo para crear su archivo de configuración local:

    ```bash
    cp backend/.env.example backend/.env
    ```

2.  **Revisar las variables de entorno (Opcional):**
    Abra el archivo `backend/.env` con un editor de texto. Para una primera ejecución en un entorno local, los valores por defecto suelen ser suficientes. Sin embargo, es útil conocer las variables principales:

    *   `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`: Credenciales para la base de datos PostgreSQL donde se almacenarán los metadatos de MLflow y Prefect.
    *   `MLFLOW_PORT`: Puerto para acceder a la interfaz de usuario de MLflow (por defecto: `5001`).
    *   `PREFECT_PORT`: Puerto para la interfaz de usuario de Prefect (por defecto: `4200`).
    *   `FASTAPI_PORT`: Puerto en el que se ejecutará la API del backend (por defecto: `8000`).
    *   `FRONTEND_PORT`: Puerto para la aplicación web del frontend (por defecto: `3000`).

    Si alguno de los puertos por defecto ya está en uso en su sistema, puede cambiarlo en este archivo antes de iniciar la aplicación.

## 4. Inicio y Verificación de la Aplicación

Una vez configurado el entorno, puede iniciar toda la pila de servicios con un único comando.

### 4.1. Construcción e Inicio de los Servicios

Desde la raíz del proyecto (`MLOps/`), ejecute el siguiente comando en su terminal:

```bash
docker compose up -d --build
```

*   **Análisis del comando:**
    *   `docker compose up`: Lee el archivo `docker-compose.yml` e inicia todos los servicios definidos (PostgreSQL, MLflow, Prefect, FastAPI, Frontend).
    *   `-d` (o `--detach`): Ejecuta los contenedores en segundo plano (modo "detached"), liberando la terminal.
    *   `--build`: Fuerza la construcción de las imágenes de Docker desde los `Dockerfile` correspondientes (`backend/Dockerfile`, `frontend/Dockerfile`, etc.) antes de iniciarlas. Este paso es esencial en la primera ejecución o después de realizar cambios en el código fuente.

El proceso de construcción puede tardar varios minutos la primera vez, ya que Docker descargará las imágenes base y instalará todas las dependencias de software.

### 4.2. Verificación del Estado de los Servicios

Para confirmar que todos los contenedores se están ejecutando correctamente, puede usar el comando:

```bash
docker compose ps
```

Debería ver una lista de los cinco servicios (`postgres`, `mlflow`, `prefect`, `fastapi`, `frontend`) con el estado `running` o `healthy`.

### 4.3. Acceso a las Interfaces de Usuario

Si la instalación y el inicio fueron exitosos, la plataforma estará accesible a través de su navegador web. Abra las siguientes URLs para verificar cada componente:

*   **Plataforma Principal (Frontend):**
    *   URL: `http://localhost:3000`
    *   Descripción: Esta es la interfaz principal de la aplicación, desde donde podrá gestionar modelos y realizar predicciones.

*   **Backend API (Documentación Interactiva):**
    *   URL: `http://localhost:8000/docs`
    *   Descripción: FastAPI genera automáticamente una página de documentación interactiva (Swagger UI) donde se pueden explorar y probar todos los endpoints de la API.

*   **Interfaz de MLflow:**
    *   URL: `http://localhost:5001`
    *   Descripción: Aquí podrá explorar en detalle los experimentos, ejecuciones, métricas y artefactos registrados por el sistema.

*   **Interfaz de Prefect:**
    *   URL: `http://localhost:4200`
    *   Descripción: El panel de control de Prefect permite monitorear la ejecución de los flujos de trabajo (pipelines) de entrenamiento y predicción.

Si todas estas páginas cargan correctamente, la plataforma está instalada y funcionando.

### 4.4. Detener la Aplicación

Para detener todos los servicios de la plataforma, ejecute el siguiente comando desde la raíz del proyecto:

```bash
docker compose down
```

Este comando detendrá y eliminará los contenedores, pero los datos persistentes (como la base de datos y los artefactos de MLflow) se conservarán en los volúmenes de Docker, listos para la próxima vez que inicie la aplicación.

## 5. Guía de Uso de la Plataforma

Una vez que la plataforma está en funcionamiento, puede interactuar con ella a través de la interfaz de usuario principal disponible en `http://localhost:3000`.

### 5.1. Panel de Control (Dashboard)

El Dashboard es la página de inicio y el punto de entrada a las funcionalidades principales. Al acceder por primera vez, verá un resumen del estado del sistema, que inicialmente estará vacío (Figura 1). Desde aquí, puede navegar a las dos secciones principales:

*   **Gestión de Modelos (Model Management):** Para entrenar, registrar, versionar y promover modelos a producción.
*   **Predicciones (Predictions):** Para utilizar los modelos en producción y realizar inferencias sobre nuevos datos.

En la cabecera de la aplicación encontrará accesos directos a las interfaces de **MLflow** y **Prefect**, herramientas esenciales para el monitoreo avanzado del sistema.

*<p align="center">Figura 1: Vista inicial del Dashboard de la plataforma.</p>*
<p align="center">
  <img src="screenshots/dashboard empty.png" width="800">
</p>

### 5.2. Gestión del Ciclo de Vida de Modelos

Esta sección es el centro de control para todo el ciclo de vida de un modelo. Al entrar por primera vez, la página estará vacía (Figura 2).

*<p align="center">Figura 2: Página de Gestión de Modelos en su estado inicial.</p>*
<p align="center">
  <img src="screenshots/model management empty.png" width="800">
</p>

El flujo de trabajo típico es el siguiente:

**Paso 1: Iniciar un Nuevo Entrenamiento**
1.  Haga clic en el botón **"Start New Training"**. Esta acción iniciará el pipeline de entrenamiento automatizado. Por defecto, el pipeline utilizará el conjunto de datos de entrenamiento ubicado en `backend/data/processed/`.
2.  Puede monitorear el progreso de la ejecución en tiempo real accediendo a las interfaces de **Prefect** (para ver el estado de las tareas, como en la Figura 3) y **MLflow** (para ver la lista de experimentos y sus ejecuciones, como en la Figura 4).

*<p align="center">Figura 3: Monitoreo de un pipeline de entrenamiento en la interfaz de Prefect.</p>*
<p align="center">
  <img src="screenshots/prefect training flow.png" width="800">
</p>

*<p align="center">Figura 4: Lista de ejecuciones de entrenamiento en la interfaz de MLflow.</p>*
<p align="center">
  <img src="screenshots/mlflow training pipeline experiment list.png" width="800">
</p>

**Paso 2: Explorar el Historial de Experimentos**
*   Una vez finalizado el entrenamiento, la tabla **"Experiment History"** en la interfaz principal mostrará un registro de la ejecución (Figura 5).
*   Cada fila representa un "parent run" en MLflow y es expandible para mostrar los "child runs" o tareas anidadas (cargar datos, procesar, entrenar). Esto permite una trazabilidad completa, mostrando el estado y la duración de cada paso del pipeline.

*<p align="center">Figura 5: Historial de experimentos después de un entrenamiento exitoso.</p>*
<p align="center">
  <img src="screenshots/model management after training before registering.png" width="800">
</p>

**Paso 3: Registrar un Modelo**
1.  En la fila de una ejecución de entrenamiento finalizada con éxito (`successful`), haga clic en el botón **"Register Model"**.
2.  Se abrirá un modal que le pedirá un nombre para el modelo (ej. `EEG_Cognitive_Decline_Classifier`), como se ve en la Figura 6.
3.  Esta acción registra formalmente el modelo entrenado y sus artefactos en el **Model Registry** de MLflow, creando la primera versión del mismo.

*<p align="center">Figura 6: Modal para registrar un nuevo modelo.</p>*
<p align="center">
  <img src="screenshots/model management model registering.png" width="800">
</p>

**Paso 4: Gestionar Versiones del Modelo**
*   Una vez que un modelo es registrado, aparecerá en el selector desplegable en la parte superior de la página. Al seleccionarlo, la interfaz se poblará con información específica de ese modelo (Figura 7).
    *   **Current Production Model:** Muestra un resumen de la versión que está actualmente en producción (inicialmente vacía).
    *   **Training History:** Una tabla filtrada que muestra solo las ejecuciones que resultaron en una versión registrada para el modelo seleccionado.
    *   **Model Versions:** Una tabla que lista todas las versiones existentes del modelo.

*<p align="center">Figura 7: Vista de un modelo registrado antes de promover una versión a producción.</p>*
<p align="center">
  <img src="screenshots/model management registered model no production version.png" width="800">
</p>

**Paso 5: Promover un Modelo a Producción**
1.  En la tabla **"Model Versions"**, localice la versión que desea desplegar para el servicio de inferencia.
2.  Haga clic en el botón **"Promote to Production"**. Se abrirá un modal de confirmación (Figura 8).
3.  Esta acción asigna el alias `"production"` a esa versión específica en el registro de MLflow. El modelo con este alias será el que se utilice por defecto en la sección de Predicciones. La interfaz se actualizará para reflejar el nuevo estado (Figura 9).

*<p align="center">Figura 8: Confirmación para promover un modelo a producción.</p>*
<p align="center">
  <img src="screenshots/model management promoting to production.png" width="800">
</p>

*<p align="center">Figura 9: Estado final con un modelo registrado y una versión en producción.</p>*
<p align="center">
  <img src="screenshots/model management model registered with production version.png" width="800">
</p>

### 5.3. Realización de Predicciones

Este módulo permite utilizar un modelo en producción para realizar inferencias sobre nuevos datos. Al acceder por primera vez, la página se mostrará vacía (Figura 10).

*<p align="center">Figura 10: Página de Predicciones en su estado inicial.</p>*
<p align="center">
  <img src="screenshots/predictions empty.png" width="800">
</p>

**Paso 1: Preparar y Cargar los Datos**
1.  Con la configuración actual el sistema está diseñado para procesar datos crudos de EEG en formato **BIDS (Brain Imaging Data Structure)**. Comprima la carpeta de su dataset BIDS en un único archivo `.zip`.
2.  Utilice el componente de carga de archivos en la interfaz para subir su archivo `.zip` al servidor. Una vez subido, el contenido del archivo se descomprimirá y aparecerá en el explorador de archivos interactivo (Figura 11).

*<p align="center">Figura 11: Vista después de cargar un archivo de datos para predecir.</p>*
<p align="center">
  <img src="screenshots/predictions before prediction.png" width="800">
</p>

**Paso 2: Seleccionar Datos y Ejecutar la Predicción**
1.  Navegue por la estructura de carpetas y seleccione el archivo o la carpeta raíz del sujeto sobre el que desea realizar la inferencia.
2.  Haga clic en el botón **"Make Prediction"**.
3.  Se abrirá un modal donde deberá seleccionar el modelo registrado que desea utilizar (Figura 12). Típicamente, seleccionará el modelo que previamente promovió a producción.

*<p align="center">Figura 12: Modal para seleccionar el modelo y ejecutar una predicción.</p>*
<p align="center">
  <img src="screenshots/prediction making a prediction.png" width="800">
</p>

**Paso 3: Consultar el Historial de Predicciones**
*   La predicción se ejecuta de forma asíncrona en el backend. Puede monitorear su progreso en las interfaces de **Prefect** (Figura 13) y **MLflow** (Figura 14).
*   Una vez completada, la tabla **"Prediction History"** en la interfaz principal se actualizará para mostrar la nueva ejecución con su estado, fecha y resultado (Figura 15).
*   Cada fila es expandible para mostrar detalles adicionales, como los datos de entrada preprocesados que se usaron para la inferencia (Figura 16).
*   Para fines de auditoría, cada predicción también se registra como una ejecución en un experimento dedicado (`Model_Predictions`) en MLflow, donde se pueden consultar todos sus detalles (Figura 17).

*<p align="center">Figura 13: Monitoreo de un pipeline de predicción en Prefect.</p>*
<p align="center">
  <img src="screenshots/prefect prediction flow.png" width="800">
</p>

*<p align="center">Figura 14: Lista de ejecuciones de predicción en MLflow.</p>*
<p align="center">
  <img src="screenshots/mlflow model predictions list.png" width="800">
</p>

*<p align="center">Figura 15: Historial de predicciones con un resultado visible.</p>*
<p align="center">
  <img src="screenshots/predictions after prediction.png" width="800">
</p>

*<p align="center">Figura 16: Modal que muestra los datos de entrada de una predicción anterior.</p>*
<p align="center">
  <img src="screenshots/prediction input details.png" width="800">
</p>

*<p align="center">Figura 17: Detalles de una ejecución de predicción en MLflow.</p>*
<p align="center">
  <img src="screenshots/mlflow prediction details.png" width="800">
</p>

### 6. Personalización y Desarrollo Avanzado

Como se mencionó en la introducción, el corazón de la plataforma es un "meta-pipeline" de entrenamiento genérico y reutilizable. El flujo de orquestación, definido en `backend/src/flows/training_flow.py`, no contiene lógica de negocio específica de un modelo. En su lugar, está diseñado para recibir tres funciones intercambiables (*pluggable*) que definen el ciclo de vida del ML:

1.  Una función de **carga de datos** (`load_data_func`)
2.  Una función de **preprocesamiento de datos** (`process_data_func`)
3.  Una función de **entrenamiento de modelo** (`train_model_func`)

Este diseño permite a los desarrolladores adaptar la plataforma para entrenar prácticamente cualquier tipo de modelo (Scikit-learn, TensorFlow, PyTorch, etc.) sobre cualquier tipo de datos, simplemente implementando sus propias funciones y conectándolas al orquestador.

#### 6.1. Guía para Implementar un Pipeline de Entrenamiento Personalizado

Para reemplazar el pipeline de ejemplo por uno propio, siga estos pasos:

**Paso 1: Preparar los Datos y Crear un Nuevo Script de Entrenamiento**

1.  **Coloque sus datos:** Agregue su conjunto de datos de entrenamiento en una subcarpeta dentro de `backend/data/`. Por ejemplo: `backend/data/my_custom_data/`.
2.  **Cree un nuevo script:** Dentro de la carpeta `backend/src/custom_scripts/`, cree un nuevo archivo Python. Por ejemplo: `my_training_script.py`. Este archivo contendrá la lógica de su modelo.

**Paso 2: Implementar las Funciones Personalizadas**

Dentro de su nuevo script (`my_training_script.py`), debe definir tres funciones que cumplan con un "contrato" específico para que el orquestador pueda ejecutarlas correctamente.

1.  **Función `load_data`:**
    *   **Propósito:** Cargar sus datos desde un archivo y devolverlos como un DataFrame de pandas.
    *   **Contrato:** Debe devolver un `pandas.DataFrame`.
    *   **Ejemplo:**
        ```python
        import pandas as pd

        def load_data():
            # Lógica para cargar tus datos
            file_path = '/app/data/my_custom_data/my_dataset.csv'
            df = pd.read_csv(file_path)
            return df
        ```

2.  **Función `process_data`:**
    *   **Propósito:** Realizar el preprocesamiento, limpieza o extracción de características.
    *   **Contrato:** Debe aceptar un `pandas.DataFrame` como entrada y devolver un `pandas.DataFrame` como salida.
    *   **Ejemplo:**
        ```python
        import pandas as pd

        def process_data(data: pd.DataFrame):
            # Lógica de preprocesamiento
            processed_df = data.dropna()
            return processed_df
        ```

3.  **Función `train_model`:**
    *   **Propósito:** Entrenar el modelo, evaluar su rendimiento y devolver el objeto del modelo junto con sus métricas.
    *   **Contrato:** Debe aceptar un `pandas.DataFrame` y devolver una tupla que contenga `(model_object, metrics_dict)`. El diccionario de métricas **debe** incluir las claves requeridas por el pipeline para el registro en MLflow.
    *   **Ejemplo:**
        ```python
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        def train_model(data: pd.DataFrame):
            # Asumir que la última columna es el target
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # Entrenar el modelo
            model = RandomForestClassifier(n_estimators=100)
            model.fit(X_train, y_train)

            # Evaluar métricas
            preds = model.predict(X_test)
            accuracy = accuracy_score(y_test, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average='macro')

            # Crear el diccionario de métricas (¡claves obligatorias!)
            metrics = {
                'accuracy': accuracy,
                'macro_avg_precision': precision,
                'macro_avg_recall': recall,
                'macro_avg_f1_score': f1,
                # Puedes añadir más métricas si lo deseas
            }

            return model, metrics
        ```

**Paso 3: Conectar el Nuevo Pipeline**

El último paso es indicarle al sistema que utilice sus nuevas funciones en lugar de las predeterminadas.

1.  Abra el archivo `backend/src/services/ml_model_service.py`.
2.  Localice la función `run_ml_training_pipeline`.
3.  Modifique las importaciones para que apunten a su nuevo script y cambie las funciones que se pasan al `ml_pipeline_flow`.

    ```python
    # En backend/src/services/ml_model_service.py

    # ... otras importaciones ...
    from ..flows.training_flow import ml_pipeline_flow
    # from ..custom_scripts.training_script import load_data, process_data, train_model # Comentar o eliminar la línea original

    # Importar las nuevas funciones
    from ..custom_scripts.my_training_script import load_data, process_data, train_model

    def run_ml_training_pipeline():
        """Runs the ML training pipeline."""
        print("Starting the CUSTOM ML training pipeline flow via service...")
        flow_state = ml_pipeline_flow(
            load_data_func=load_data,      # Tu nueva función
            load_data_args=(),
            load_data_kwargs={},
            process_data_func=process_data,  # Tu nueva función
            process_data_args=(),
            process_data_kwargs={},
            train_model_func=train_model,    # Tu nueva función
            train_model_args=(),
            train_model_kwargs={}
        )
        print("CUSTOM ML training pipeline flow finished in service.")
        return flow_state
    ```

**Paso 4: Reconstruir la Imagen de Docker**

Después de realizar cambios en el código del backend, es necesario reconstruir la imagen de Docker del servicio `fastapi` para que los cambios surtan efecto.

Ejecute el siguiente comando desde la raíz del proyecto:

```bash
docker compose up -d --build fastapi
```

¡Listo! La próxima vez que presione el botón **"Start New Training"** en la interfaz de usuario, la plataforma ejecutará su pipeline personalizado.

#### 6.2. Personalización del Pipeline de Predicción

El pipeline de predicción sigue un principio similar. Si necesita procesar datos de entrada crudos con un formato diferente al BIDS, deberá modificar la función de preprocesamiento que utiliza el flujo de predicción.

1.  **Función a modificar:** `process_data` en el archivo `backend/src/custom_scripts/data_preprocessing_script.py`.
2.  **Contrato:** La función debe aceptar una ruta a los datos de entrada y devolver un `pandas.DataFrame` con las características que el modelo espera recibir.
3.  **Reconstrucción:** Al igual que con el pipeline de entrenamiento, deberá reconstruir la imagen con `docker compose up -d --build fastapi` después de realizar los cambios.

