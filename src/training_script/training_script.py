import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usar backend 'Agg' para no necesitar interfaz gráfica
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix
from .utils.training_functions import *
from datetime import datetime
import os
from dotenv import load_dotenv

# Define no-op loggers directly in this script
def _noop(*args, **kwargs): pass

noop_loggers = {
    "log_param": _noop,
    "log_metric": _noop,
    "log_artifact": _noop,
    "log_model": _noop, # Include log_model for completeness, though handled by flow
}

# Load environment variables from .env file
load_dotenv()


TIME_NOW = datetime.now().strftime('%Y%m%d_%H%M%S')

CLASS_NAMES = ['ACr', 'HC']
NEURO = 'neuroHarmonize'
NAME = 'G1'
SPACE = 'ic'
STR_RATIO = '2to1'
PROJECT_PATH = os.path.abspath('/Users/imeag/Documents/udea/trabajoDeGrado/MLOps') 
DATA_FILE_PATH = os.path.join(PROJECT_PATH, 'data', 'processed', f'Data_integration_ic_{NEURO}_{NAME}.feather')
PATH_SAVE_BASE = os.path.join(PROJECT_PATH, 'Experimenting', 'Results', TIME_NOW)
PATH_SAVE_CORRELATION_MAP = os.path.join(PATH_SAVE_BASE, 'correlation_map')

modelos1 = {}
acc_per_feature1 = []
std_per_feature1 = []

def load_data(*args, **kwargs):
    """Loads data from feather file and logs initial info."""
    # Get logger functions from kwargs, default to no-op if not provided
    loggers = kwargs.get("mlflow_loggers", noop_loggers)
    log_param = loggers.get("log_param", _noop)
    log_metric = loggers.get("log_metric", _noop)
    log_artifact = loggers.get("log_artifact", _noop)
    
    log_param("data_file_path", DATA_FILE_PATH)
    data = pd.read_feather(DATA_FILE_PATH)
    # Example: Log parameters specific to loading if needed
    # log_param("initial_rows", data.shape[0]) 
    return data
  
def process_data(data, *args, **kwargs):
    """Processes data and splits it into training and testing sets."""
    # Get logger functions from kwargs, default to no-op if not provided
    loggers = kwargs.get("mlflow_loggers", noop_loggers)
    log_param = loggers.get("log_param", _noop)
    log_metric = loggers.get("log_metric", _noop)
    log_artifact = loggers.get("log_artifact", _noop)

    # Eliminación de columnas con datos faltantes para data
    col_del1 = pd.DataFrame()
    for column in data.columns:
        if data[column].isna().sum() != 0:
            col_del1[column] = [data[column].isna().sum()]
            data.drop(column, axis=1, inplace=True)
            
    # Se mapean las clases para data
    clases_mapeadas1 = {label: idx for idx, label in enumerate(np.unique(data['group']))}
    data.loc[:, 'group'] = data.loc[:, 'group'].map(clases_mapeadas1)
    log_param("class_mapping", str(clases_mapeadas1))
    
    # Se elimina la columna, para ponerla al final para data
    target1 = data.pop('group')
    data.insert(len(data.columns), target1.name, target1)
    data['group'] = pd.to_numeric(data['group'])
    
    data.select_dtypes('O')
    data.groupby(by='sex').describe().T
    sexo_mapeado1 = {label: idx for idx, label in enumerate(np.unique(data['sex']))}
    data.loc[:, 'sex'] = data.loc[:, 'sex'].map(sexo_mapeado1)
    log_param("sex_mapping", str(sexo_mapeado1))
    
    # data pasa a ser el arreglo únicamente con los datos númericos
    numerics1 = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    data = data.select_dtypes(include=numerics1)
    data.shape

    data = mapa_de_correlacion(data, PATH_SAVE_CORRELATION_MAP, STR_RATIO)
    # TODO: Log the correlation map artifact

    return data
  
def train_model(data, *args, **kwargs):
  # Get logger functions from kwargs, default to no-op if not provided
  loggers = kwargs.get("mlflow_loggers", noop_loggers)
  log_param = loggers.get("log_param", _noop)
  log_metric = loggers.get("log_metric", _noop)
  log_artifact = loggers.get("log_artifact", _noop)
  # log_model is handled by the flow task
  # log_model = loggers.get("log_model", _noop) 

  X1 = data.values[:, :-1]  # La ultima posicion es el grupo, por eso se elimina
  y1 = data.values[:, -1]
  print(f'Data Shape: X1 (features) shape: {X1.shape}')
  print(f'Data Shape: y1 (target) shape: {y1.shape}')

  TEST_SIZE=0.2 # Test de 20%
  RANDOM_STATE=1 # Semilla
  X_train1, X_test1, y_train1, y_test1 = train_test_split(
      X1,
      y1,
      test_size=TEST_SIZE,
      random_state=RANDOM_STATE,
      stratify=data.values[:, -1])
  
  # Log split info
  log_param("test_size", TEST_SIZE)
  log_param("random_state", RANDOM_STATE)
  log_param("n_train_samples", X_train1.shape[0])
  log_param("n_test_samples", X_test1.shape[0])
  
  random_grid1 = grid_search()
  rf_random1 = randomFo(random_grid1, X_train1, y_train1)
  best_selected1 = rf_random1.best_estimator_
  params1 = rf_random1.best_params_
  
  # Log best parameters
  for param, value in params1.items():
      log_param(f"best_{param}", value)

  # Guardar mejores características
  feat1 = pd.DataFrame()
  sorted_names1 = []
  nombres_columnas1 = data.columns[:-1]
  features_scores1 = best_selected1.feature_importances_
  index1 = np.argsort(features_scores1)[::-1]
  feat1 = primeras_carateristicas(X_train1, sorted_names1, nombres_columnas1, features_scores1, feat1, index1, PATH_SAVE_BASE, STR_RATIO)

  # Log feature importance
  #for i, (feature, importance) in enumerate(zip(nombres_columnas1, features_scores1)):
      #mlflow.log_metric(f"feature_importance_{feature}", float(importance))

  curva_de_aprendizaje(sorted_names1, data, best_selected1, X_train1, y_train1, modelos1, acc_per_feature1, std_per_feature1, PATH_SAVE_BASE, STR_RATIO)
  
  GS_fitted1 = best_selected1.fit(X_train1, y_train1)
  modelos1['GridSerach'] = GS_fitted1
  
  predicted1 = GS_fitted1.predict(X_test1)
  predicted_proba = GS_fitted1.predict_proba(X_test1)[:, 1]  # Probabilidades de la clase positiva
  print(
      f"Results: Classification report for classifier {GS_fitted1}:\n"
      f"{metrics.classification_report(y_test1, predicted1)}\n"
  )
  dataframe_metrics1 = metrics.classification_report(y_test1, predicted1, output_dict=True)
  dataframe_metrics1 = pd.DataFrame(dataframe_metrics1).T
  scores1 = cross_val_score(
      estimator=GS_fitted1,
      X=X_train1,
      y=y_train1,
      cv=10,
      n_jobs=-1
  )
  print('CV Results: Accuracy scores: %s' % scores1)
  print('CV Results: Mean Accuracy: %.3f +/- %.3f' % (np.mean(scores1), np.std(scores1)))

  # Log cross-validation metrics
  log_metric("cv_mean_accuracy", float(np.mean(scores1)))
  log_metric("cv_std_accuracy", float(np.std(scores1)))

  acc_per_feature1.append(np.mean(scores1))
  std_per_feature1.append(np.std(scores1))

  pos_model1 = np.argsort(acc_per_feature1)[-1]
  best_model1 = list(modelos1.keys())[pos_model1]
  best_features1 = sorted_names1[:pos_model1]
  mi_path1 = os.path.join(PATH_SAVE_BASE, 'best_params1.txt')
  with open(mi_path1, 'w') as f:
      for i in params1:
          f.write(f"{i}\n")
  log_artifact(mi_path1, "parameters") # Log the saved parameters file
  
  # Log best model info
  log_param("best_model_name", best_model1) # Changed key slightly for clarity
  log_param("n_best_features", len(best_features1))
  
  title = f'validation_GridSearch.png'
  palette1 = ["#8AA6A3", "#127369"]

  curva_validacion3(GS_fitted1, X_train1, y_train1, title, palette1, STR_RATIO)
  plt.grid()
  fig = plt.gcf()
  fig.savefig(os.path.join(PATH_SAVE_BASE, title), bbox_inches='tight')
  plt.close()

  # Log artifacts
  # mlflow.log_artifact(os.path.join(PATH_SAVE_BASE, title))
  # mlflow.log_artifact(mi_path1)
  # mlflow.log_artifact(path_excel1_1)
  # mlflow.log_artifact(path_excel1_2)
  # mlflow.log_artifact(path_excel1_3) #FIXME: this is causing an error

  acc1, std1, fbest_model1, input_best_index1 = features_best3(best_features1, best_selected1, data.iloc[:, :-1], X_train1, y_train1, PATH_SAVE_BASE)
  
  # return the model trained
  return GS_fitted1
  
