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