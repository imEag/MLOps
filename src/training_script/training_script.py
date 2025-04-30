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
from src.utils.training_functions import *
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

CLASS_NAMES = ['ACr', 'HC']
NEURO = 'neuroHarmonize'
NAME = 'G1'
SPACE = 'ic'
STR_RATIO = '2to1'
PROJECT_PATH = os.path.abspath('/Users/imeag/Documents/udea/trabajoDeGrado/MLOps') 
DATA_FILE_PATH = os.path.join(PROJECT_PATH, 'data', 'processed', f'Data_integration_ic_{NEURO}_{NAME}.feather')
PATH_SAVE_BASE = os.path.join(PROJECT_PATH, 'Experimenting', 'Results') 

def load_data():
    """Loads data from feather file and logs initial info."""
    data = pd.read_feather(DATA_FILE_PATH)
    return data
