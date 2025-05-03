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

def load_data():
    """Loads data from feather file and logs initial info."""
    data = pd.read_feather(DATA_FILE_PATH)
    return data
  
def process_data(data):
    """Processes data and splits it into training and testing sets."""
    
    # Eliminación de columnas con datos faltantes para data
    col_del1 = pd.DataFrame()
    for column in data.columns:
        if data[column].isna().sum() != 0:
            col_del1[column] = [data[column].isna().sum()]
            data.drop(column, axis=1, inplace=True)
            
    # Se mapean las clases para data
    clases_mapeadas1 = {label: idx for idx, label in enumerate(np.unique(data['group']))}
    data.loc[:, 'group'] = data.loc[:, 'group'].map(clases_mapeadas1)
    #mlflow.log_param("class_mapping", str(clases_mapeadas1))
    
    # Se elimina la columna, para ponerla al final para data
    target1 = data.pop('group')
    data.insert(len(data.columns), target1.name, target1)
    data['group'] = pd.to_numeric(data['group'])
    
    data.select_dtypes('O')
    data.groupby(by='sex').describe().T
    sexo_mapeado1 = {label: idx for idx, label in enumerate(np.unique(data['sex']))}
    data.loc[:, 'sex'] = data.loc[:, 'sex'].map(sexo_mapeado1)
    #mlflow.log_param("sex_mapping", str(sexo_mapeado1))
    
    # data pasa a ser el arreglo únicamente con los datos númericos
    numerics1 = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    data = data.select_dtypes(include=numerics1)
    data.shape

    data = mapa_de_correlacion(data, PATH_SAVE_CORRELATION_MAP, STR_RATIO)

    return data
  
def train_model(data):
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
  #mlflow.log_param("test_size", TEST_SIZE)
  #mlflow.log_param("random_state", RANDOM_STATE)
  #mlflow.log_param("n_train_samples", X_train1.shape[0])
  #mlflow.log_param("n_test_samples", X_test1.shape[0])
  
  random_grid1 = grid_search()
  rf_random1 = randomFo(random_grid1, X_train1, y_train1)
  best_selected1 = rf_random1.best_estimator_
  params1 = rf_random1.best_params_
  
  # Log best parameters
  #for param, value in params1.items():
      #mlflow.log_param(f"best_{param}", value)

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
  
  #return the model trained
  return GS_fitted1