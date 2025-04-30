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
import joblib
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- MLflow Configuration ---
MLFLOW_PORT = os.getenv('MLFLOW_PORT', '5000') # Default to 5000 if not set
mlflow.set_tracking_uri(f"http://localhost:{MLFLOW_PORT}")

TIME_NOW = datetime.now().strftime('%Y%m%d_%H%M%S')

# TODO: Do a new refactor of the script manually, step by step. Based on this one and the original script.
# TODO: Review if the script is working as expected. Specialy the:
# - best_features.txt
# TODO: Convert this script to a prefect flow
# TODO: Add a prefect flow to log the model to MLflow
# TODO: Review and add all mlflow parameters


# --- Helper Functions for Refactoring ---

def setup_mlflow(neuro, name, space, var1):
    """Sets up the MLflow experiment and starts a new run."""
    # experiment_name = f"{neuro}_{name}_{space}_{var1}" # Experiment is set outside now
    # mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=f"training_{TIME_NOW}", nested=True) # Start as nested run
    print(f"MLflow Training Run ID: {run.info.run_id}") # Log nested run ID
    mlflow.set_tag("neuro", neuro)
    mlflow.set_tag("name", name)
    mlflow.set_tag("space", space)
    mlflow.set_tag("ratio", var1)
    # return run.info.run_id # No longer need to return run_id

def define_paths(path_save, space, name, var1):
    """Defines and creates necessary directory paths."""
    path_excel1 = os.path.join(path_save, f'tables/ML/{space}/{name}')
    path_excel1_1 = os.path.join(path_excel1, f'describe_all_{var1}.xlsx')
    path_excel1_2 = os.path.join(path_excel1, f'describe_{var1}.xlsx')
    path_excel1_3 = os.path.join(path_excel1, f'features_{var1}.xlsx') # Path defined but file not generated/logged
    path_plot = os.path.join(path_save, 'graphics', 'ML', space, f'{name}_{var1}') # Simplified path_plot

    os.makedirs(path_plot, exist_ok=True)
    os.makedirs(path_excel1, exist_ok=True)

    return path_excel1_1, path_excel1_2, path_excel1_3, path_plot


def load_and_log_initial_data(data_path):
    """Loads data from feather file and logs initial info."""
    data = pd.read_feather(data_path)
    print(f'Info: Initial dataset shape: {data.shape[0]} samples | {data.shape[1]} features')
    mlflow.log_param("initial_n_samples", data.shape[0])
    mlflow.log_param("initial_n_features", data.shape[1])

    for group in data['group'].unique():
        mlflow.log_metric(f"initial_n_samples_{group}", (data['group'] == group).sum())

    return data


def preprocess_data(data, path_excel1_1, path_excel1_2, log_artifacts=True):
    """Preprocesses the data: handles missing values, maps categorical features."""
    # Save Excel description files
    data.describe().T.to_excel(path_excel1_1)
    data.groupby(by='group').describe().T.to_excel(path_excel1_2)
    if log_artifacts:
        mlflow.log_artifact(path_excel1_1)
        mlflow.log_artifact(path_excel1_2)
        # mlflow.log_artifact(path_excel1_3) # FIXME: This file isn't generated here

    # Handle missing values
    col_del1 = pd.DataFrame()
    for column in data.columns:
        if data[column].isna().sum() != 0:
            col_del1[column] = [data[column].isna().sum()]
            data.drop(column, axis=1, inplace=True)
    
    # Log dropped columns
    if col_del1.empty:
        mlflow.log_param("dropped_columns", "None")
    else:
        mlflow.log_param("dropped_columns", col_del1.to_dict(orient='records'))
    

    # Map 'group'
    clases_mapeadas1 = {label: idx for idx, label in enumerate(np.unique(data['group']))}
    data.loc[:, 'group'] = data.loc[:, 'group'].map(clases_mapeadas1)
    mlflow.log_param("class_mapping", str(clases_mapeadas1))

    # Reorder 'group' column
    target1 = data.pop('group')
    data.insert(len(data.columns), target1.name, target1)
    data['group'] = pd.to_numeric(data['group'])

    # Map 'sex'
    sexo_mapeado1 = {label: idx for idx, label in enumerate(np.unique(data['sex']))}
    data.loc[:, 'sex'] = data.loc[:, 'sex'].map(sexo_mapeado1)
    mlflow.log_param("sex_mapping", str(sexo_mapeado1))

    # Select numeric types
    numerics1 = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    data_numeric = data.select_dtypes(include=numerics1)
    print(f'Info: Shape after preprocessing and numeric selection: {data_numeric.shape}')
    mlflow.log_param("n_features_numeric", data_numeric.shape[1])

    return data_numeric, clases_mapeadas1, sexo_mapeado1


def generate_correlation_map_and_reduce(data, path_plot, var1, log_artifacts=True):
    """Generates correlation map, reduces features based on correlation, and logs."""
    data_reduced = mapa_de_correlacion(data, path_plot, var1) # This function already saves plots
    if log_artifacts:
        mlflow.log_artifact(os.path.join(path_plot, 'correlation_before.png'))
        mlflow.log_artifact(os.path.join(path_plot, 'correlation_after.png'))
        mlflow.log_artifact(os.path.join(path_plot, f'Data_integration_corr.feather'))
    print(f'Info: Shape after correlation reduction: {data_reduced.shape}')
    mlflow.log_param("n_features_after_corr", data_reduced.shape[1])
    return data_reduced


def split_data(data, test_size=0.2, random_state=1):
    """Splits data into training and testing sets."""
    X = data.values[:, :-1]
    y = data.values[:, -1]
    print(f'Data Shape: X (features) shape: {X.shape}')
    print(f'Data Shape: y (target) shape: {y.shape}')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y # Keep original stratification logic
    )

    # Log split info
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("n_train_samples", X_train.shape[0])
    mlflow.log_param("n_test_samples", X_test.shape[0])

    return X_train, X_test, y_train, y_test


def tune_hyperparameters(X_train, y_train):
    """Performs Randomized Search CV for RandomForest."""
    random_grid = grid_search() # From utils
    rf_random = randomFo(random_grid, X_train, y_train) # From utils
    best_estimator = rf_random.best_estimator_
    best_params = rf_random.best_params_

    # Log best parameters
    for param, value in best_params.items():
        mlflow.log_param(f"best_{param}", value)

    return best_estimator, best_params


def calculate_and_log_feature_importance(best_estimator, X_train, column_names, path_plot, var1, log_artifacts=True):
    """Calculates feature importance, saves plot, and logs metrics."""
    feat_df = pd.DataFrame()
    sorted_names = []
    features_scores = best_estimator.feature_importances_
    index = np.argsort(features_scores)[::-1]

    # Assuming `primeras_carateristicas` calculates, prints, plots and returns the dataframe
    # Note: `primeras_carateristicas` also saves a plot `features_table_plot_all.png`
    feat_df = primeras_carateristicas(X_train, sorted_names, column_names, features_scores, feat_df, index, path_plot, var1)
    # `sorted_names` is modified in place by `primeras_carateristicas`

    # Log feature importance metrics
    for i, (feature, importance) in enumerate(zip(column_names, features_scores)):
         # Ensure importance is a standard float before logging
        try:
            importance_float = float(importance)
            mlflow.log_metric(f"feature_importance_{feature}", importance_float)
        except ValueError:
            print(f"Warning: Could not convert importance for feature '{feature}' to float: {importance}")


    # Log feature importance plot
    if log_artifacts:
        plot_path = os.path.join(path_plot, 'features_table_plot_all.png')
        if os.path.exists(plot_path):
             mlflow.log_artifact(plot_path)
        else:
             print(f"Warning: Feature importance plot not found at {plot_path}")

    # Save features dataframe (if needed, originally path_excel1_3)
    features_excel_path = os.path.join(os.path.dirname(path_plot), f'features_{var1}.xlsx') # Save it near plots
    feat_df.T.to_excel(features_excel_path, header=False)
    if log_artifacts:
        mlflow.log_artifact(features_excel_path)


    return sorted_names, feat_df, features_scores # Return scores as well


# --- Original exec1 logic, refactored ---
def exec1(neuro, name, space, path_save, data_path, var1, class_names, model=None):
    """
    Refactored training logic corresponding to the original exec1.
    Handles MLflow setup, data loading, preprocessing, training, initial evaluation,
    and feature analysis.
    """
    # run_id = setup_mlflow(neuro, name, space, var1) # setup_mlflow no longer returns run_id
    setup_mlflow(neuro, name, space, var1) # Call setup for nested run and tags

    # Define paths
    path_excel1_1, path_excel1_2, path_excel1_3, path_plot = define_paths(path_save, space, name, var1)


    if model is None:
        # Load data
        data_initial = load_and_log_initial_data(data_path)

        # Preprocess data
        data_processed, clases_mapeadas1, _ = preprocess_data(data_initial.copy(), path_excel1_1, path_excel1_2) # Use copy to avoid modifying original df

        # Correlation Analysis
        data_reduced = generate_correlation_map_and_reduce(data_processed, path_plot, var1)

        # Get column names *before* converting to numpy
        nombres_columnas1 = data_reduced.columns[:-1] # Exclude target column 'group'

        # Split data
        X1 = data_reduced.values[:, :-1]
        y1 = data_reduced.values[:, -1]
        X_train1, X_test1, y_train1, y_test1 = split_data(data_reduced, test_size=0.2, random_state=1)

        # --- Model Training and Initial Evaluation ---
        modelos1 = {} # To store models from learning curve
        acc_per_feature1 = [] # To store accuracy from learning curve
        std_per_feature1 = [] # To store std dev from learning curve

        # Hyperparameter Tuning
        best_selected1, params1 = tune_hyperparameters(X_train1, y_train1)

        # Feature Importance
        sorted_names1, feat1_df, features_scores1 = calculate_and_log_feature_importance(
            best_selected1, X_train1, nombres_columnas1, path_plot, var1
        )

        # Learning Curve by Feature Count (using original function)
        # This function modifies `modelos1`, `acc_per_feature1`, `std_per_feature1` in place
        # and saves a plot 'features_plot_all.png'
        curva_de_aprendizaje(sorted_names1, data_reduced, best_selected1, X_train1, y_train1, modelos1, acc_per_feature1, std_per_feature1, path_plot, var1)
        mlflow.log_artifact(os.path.join(path_plot, 'features_plot_all.png'))


        # Train final model with best params (on full training set)
        # Note: best_selected1 is already the estimator with best params found by RandomizedSearchCV
        # Re-fitting it here on the full training set is standard practice.
        GS_fitted1 = best_selected1.fit(X_train1, y_train1)
        # modelos1['GridSerach'] = GS_fitted1 # This seems redundant if GS_fitted1 is the main model now

        # Initial prediction and metrics on test set
        predicted1 = GS_fitted1.predict(X_test1)
        predicted_proba = GS_fitted1.predict_proba(X_test1)[:, 1] # Probabilities of the class positiva

        print(
            f"Results: Classification report for classifier {GS_fitted1}:"
            f"{metrics.classification_report(y_test1, predicted1)}"
        )
        # TODO: Log classification report (e.g., as text artifact or individual metrics)
        report_dict = metrics.classification_report(y_test1, predicted1, output_dict=True)
        # Log key metrics from the report
        mlflow.log_metric("test_accuracy", float(report_dict['accuracy']))
        mlflow.log_metric("test_precision_weighted", float(report_dict['weighted avg']['precision']))
        mlflow.log_metric("test_recall_weighted", float(report_dict['weighted avg']['recall']))
        mlflow.log_metric("test_f1_weighted", float(report_dict['weighted avg']['f1-score']))


        # Cross-validation on the training set with the best estimator
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
        mlflow.log_metric("cv_mean_accuracy", float(np.mean(scores1)))
        mlflow.log_metric("cv_std_accuracy", float(np.std(scores1)))

        # This was part of curva_de_aprendizaje logic, seems misplaced here
        # acc_per_feature1.append(np.mean(scores1))
        # std_per_feature1.append(np.std(scores1))

        # --- Selecting Best Model/Features based on Learning Curve ---
        # This section relies on the `acc_per_feature1` list populated by `curva_de_aprendizaje`
        if acc_per_feature1: # Check if list is not empty
             pos_model1 = np.argsort(acc_per_feature1)[-1] # Index of best accuracy
             best_model_name1 = list(modelos1.keys())[pos_model1] # Name like 'number_features_X'
             num_best_features = int(best_model_name1.split('_')[-1]) # Extract number of features X
             best_features1 = sorted_names1[:num_best_features] # Select the top X features
             # Retrieve the actual model trained with these features from the `modelos1` dict
             fbest_model1 = modelos1[best_model_name1]
             input_best_index1 = [nombres_columnas1.get_loc(c) for c in best_features1 if c in nombres_columnas1]


             print(f"Info: Best model identified from learning curve: {best_model_name1} with {len(best_features1)} features.")
             mlflow.log_param("best_model_from_lc", best_model_name1)
             mlflow.log_param("n_best_features_from_lc", len(best_features1))

             # Save best feature names
             mi_path1 = os.path.join(path_plot, 'best_features.txt') # Renamed from best_params1.txt
             with open(mi_path1, 'w') as f:
                for feature in best_features1:
                     f.write(f"{feature}")
             mlflow.log_artifact(mi_path1)

             # Generate Validation Curve for the *final selected model* (fbest_model1) using its features
             title_val_curve = f'validation_curve_best_{len(best_features1)}_features.png'
             palette1 = ["#8AA6A3", "#127369"]
             # We need to use the subset of X_train1 corresponding to best_features1
             X_train_best_features = X_train1[:, input_best_index1]
             curva_validacion3(fbest_model1, X_train_best_features, y_train1, title_val_curve, palette1, var1)
             plt.grid()
             fig = plt.gcf()
             val_curve_path = os.path.join(path_plot, title_val_curve)
             fig.savefig(val_curve_path, bbox_inches='tight')
             plt.close()
             mlflow.log_artifact(val_curve_path)

             # The original code called `features_best3` here.
             # `features_best3` seems to *re-run* learning curve analysis specifically on the `best_features1` list.
             # This might be redundant if `fbest_model1` and `input_best_index1` are already determined.
             # Let's replicate the call but analyze if it's truly needed or just for plotting.
             # `features_best3` saves 'features_plot_best.png'
             print("Running features_best3 analysis...")
             acc_best, std_best, _, _ = features_best3(best_features1, fbest_model1, data_reduced.iloc[:, :-1], X_train1, y_train1, path_plot)
             if acc_best: # Check if returned values are valid
                 mlflow.log_artifact(os.path.join(path_plot, 'features_plot_best.png'))
                 final_acc = acc_best[-1] # Get the accuracy using all best_features
                 final_std = std_best[-1] # Get the std dev using all best_features
                 mlflow.log_metric("final_lc_best_accuracy", float(final_acc))
                 mlflow.log_metric("final_lc_best_std", float(final_std))
             else:
                  print("Warning: features_best3 did not return valid results.")
                  final_acc = None
                  final_std = None

        else:
             print("Warning: Learning curve analysis did not produce results (acc_per_feature1 is empty). Cannot select best model based on features.")
             # Fallback or error handling needed? For now, proceed with GS_fitted1 as the best model.
             fbest_model1 = GS_fitted1
             input_best_index1 = list(range(X_train1.shape[1])) # Use all features
             best_features1 = list(nombres_columnas1)
             final_acc = np.mean(scores1) # Use CV results as fallback
             final_std = np.std(scores1)
             predicted_proba = GS_fitted1.predict_proba(X_test1)[:, 1] # Ensure this is set

    else:
        # --- Fine-tuning logic (if model is provided) ---
        # This part needs similar refactoring if it's intended to be used.
        # For now, keeping it less refactored based on the primary goal.
        print("Info: Fine-tuning existing model.")
        mlflow.set_tag("mode", "fine-tuning")

        # Minimal preprocessing needed? Assuming input data matches model expectations.
        # Let's reuse some preprocessing steps, adapted for fine-tuning.
        data_initial = load_and_log_initial_data(data_path) # Reuse loading
        # Apply mappings if needed? Assume data_path data matches original structure.
        clases_mapeadas1 = {label: idx for idx, label in enumerate(np.unique(data_initial['group']))} # Recalculate mapping
        data_initial.loc[:, 'group'] = data_initial.loc[:, 'group'].map(clases_mapeadas1)
        sexo_mapeado1 = {label: idx for idx, label in enumerate(np.unique(data_initial['sex']))}
        data_initial.loc[:, 'sex'] = data_initial.loc[:, 'sex'].map(sexo_mapeado1)
        numerics1 = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        data_numeric = data_initial.select_dtypes(include=numerics1)

        # Correlation Map (optional for fine-tuning, depends on scenario)
        data_reduced = generate_correlation_map_and_reduce(data_numeric, path_plot, var1, log_artifacts=False) # Avoid re-logging artifacts for tuning run maybe?

        # Split data
        X_train1, X_test1, y_train1, y_test1 = split_data(data_reduced, test_size=0.2, random_state=1) # Reuse splitting

        # Log fine-tuning split info separately
        mlflow.log_param("fine_tuning_test_size", 0.2)
        mlflow.log_param("fine_tuning_random_state", 1)
        mlflow.log_param("fine_tuning_n_train_samples", X_train1.shape[0])
        mlflow.log_param("fine_tuning_n_test_samples", X_test1.shape[0])

        # Fit the provided model
        model.fit(X_train1, y_train1)
        predicted1 = model.predict(X_test1)
        predicted_proba = model.predict_proba(X_test1)[:, 1]

        print(
            f"Results (Fine-tuning): Classification report for classifier {model}:"
            f"{metrics.classification_report(y_test1, predicted1)}"
        )
        # TODO: Log fine-tuning classification report

        scores1 = cross_val_score(
            estimator=model, X=X_train1, y=y_train1, cv=10, n_jobs=-1
        )
        print('CV Results (Fine-tuning): Mean Accuracy: %.3f +/- %.3f' % (np.mean(scores1), np.std(scores1)))

        # Log fine-tuning metrics
        mlflow.log_metric("fine_tuning_cv_mean_accuracy", float(np.mean(scores1)))
        mlflow.log_metric("fine_tuning_cv_std_accuracy", float(np.std(scores1)))

        # Assign return values for fine-tuning case
        final_acc = np.mean(scores1)
        final_std = np.std(scores1)
        fbest_model1 = model
        # Fine-tuning might not involve feature selection, assume all features used by the input model
        input_best_index1 = list(range(X_train1.shape[1]))
        # clases_mapeadas1 needs to be returned, ensure it's available

    # End the training run explicitly? Or let context manager handle it?
    # Assuming the context manager handles the end when the script finishes.

    # Return values needed by exec2
    # Ensure all paths exist in the 'else' block if needed, or return None
    if 'fbest_model1' not in locals(): fbest_model1 = None
    if 'input_best_index1' not in locals(): input_best_index1 = None
    if 'X_train1' not in locals(): X_train1 = None
    if 'y_train1' not in locals(): y_train1 = None
    if 'clases_mapeadas1' not in locals(): clases_mapeadas1 = None # Should be defined in both branches
    # var1 is input
    if 'predicted_proba' not in locals(): predicted_proba = None # Ensure defined
    if 'X_test1' not in locals(): X_test1 = None
    if 'y_test1' not in locals(): y_test1 = None
    if 'final_acc' not in locals(): final_acc = None # Renamed from acc1
    if 'final_std' not in locals(): final_std = None # Renamed from std1

    # Return path_plot as well for exec2
    return (final_acc, final_std, fbest_model1, input_best_index1, X_train1, y_train1,
            clases_mapeadas1, var1, predicted_proba, X_test1, y_test1, path_plot)

# --- Original exec2 logic (placeholder for refactoring) ---
def exec2(acc1, std1, fbest_model1, input_best_index1, X_train1, y_train1, clases_mapeadas1, path_plot, var1, predicted_proba, X_test=None, y_test=None):
    # Start evaluation run in MLflow (can be nested or separate)
    # Using nested run for now
    with mlflow.start_run(run_name=f"evaluation_{TIME_NOW}", nested=True) as run:
        print(f"MLflow Evaluation Run ID: {run.info.run_id}")
        mlflow.set_tag("phase", "evaluation")

        band = 0 # Flag to indicate if using training data for evaluation
        if X_test is None or y_test is None:
            X_test_eval = X_train1 # Use training data if no test data provided
            y_test_eval = y_train1
            band = 1
            print("Info: Evaluating on TRAINING data as no test data was provided.")
            mlflow.log_param("evaluation_data_source", "training_set")
        else:
            X_test_eval = X_test # Use provided test data
            y_test_eval = y_test
            print("Info: Evaluating on provided TEST data.")
            mlflow.log_param("evaluation_data_source", "test_set")

        # Verify input data shapes and labels for debugging/logging
        print("Data Shape (Evaluation): Training X shape:", X_train1.shape)
        print("Data Info (Evaluation): Training y labels:", np.unique(y_train1, return_counts=True))
        print("Data Shape (Evaluation): Test X shape:", X_test_eval.shape)
        print("Data Info (Evaluation): Test y labels:", np.unique(y_test_eval, return_counts=True))
        print("Info: Class mapping used for evaluation:", clases_mapeadas1)

        # --- Apply Feature Selection ---
        # Ensure input_best_index1 is valid
        if input_best_index1 is None:
             print("Error: input_best_index1 is None. Cannot select features for evaluation.")
             # Handle error appropriately, e.g., return or raise exception
             return None # Or raise ValueError("Feature indices not available for evaluation")
        
        # Check index bounds before slicing
        max_index = max(input_best_index1) if input_best_index1 else -1
        num_features_eval = X_test_eval.shape[1]
        if max_index >= num_features_eval:
            print(f"Error: Max feature index ({max_index}) is out of bounds for X_test_eval with {num_features_eval} features.")
            # Handle error
            return None

        # Apply feature selection to the evaluation data (either test or train)
        X_test_selected = X_test_eval[:, input_best_index1]
        print(f"Info: Applied feature selection. Evaluation X shape: {X_test_selected.shape}")
        mlflow.log_param("n_features_evaluated", X_test_selected.shape[1])

        # --- Make Predictions ---
        if fbest_model1 is None:
            print("Error: fbest_model1 is None. Cannot perform evaluation.")
            return None

        # Predict on selected features
        predicted_eval = fbest_model1.predict(X_test_selected)
        # Predict probabilities needed for AUC - use the same selected features
        predicted_proba_eval = fbest_model1.predict_proba(X_test_selected)[:, 1]

        # The original code had a special case for band == 1 for label conversion,
        # but metrics functions usually handle numeric labels directly. Let's verify.
        # `y_test_eval` should already be numeric due to preprocessing.
        # classes_x1 = (predicted_eval >= 0.5).astype(int) # This is redundant if predict gives 0/1

        # --- Calculate and Log Metrics ---
        try:
            # Ensure labels are integers if necessary for metric functions
            y_test_eval_int = y_test_eval.astype(int)
            predicted_eval_int = predicted_eval.astype(int)

            # Calculate standard metrics
            precision = precision_score(y_test_eval_int, predicted_eval_int)
            recall = recall_score(y_test_eval_int, predicted_eval_int)
            f1 = f1_score(y_test_eval_int, predicted_eval_int)
            # Calculate AUC using probabilities
            auc_score_eval = roc_auc_score(y_test_eval_int, predicted_proba_eval)

            print(f"Evaluation Metrics: /nPrecision: {precision} /nRecall: {recall} /nF1 Score: {f1} /nAUC: {auc_score_eval}")

            # Log final metrics to MLflow
            mlflow.log_metric("final_precision", float(precision))
            mlflow.log_metric("final_recall", float(recall))
            mlflow.log_metric("final_f1", float(f1))
            mlflow.log_metric("final_auc", float(auc_score_eval)) # Use the probability-based AUC

            # Save metrics to CSV
            metrics_dict = {'Precision': [precision], 'Recall': [recall], 'F1': [f1], 'AUC': [auc_score_eval]}
            path_metrics_csv = os.path.join(path_plot, f'metrics_ML_{var1}.csv')
            metrics_df = pd.DataFrame(metrics_dict)
            metrics_df.to_csv(path_metrics_csv, index=False)
            mlflow.log_artifact(path_metrics_csv)

        except Exception as e:
            print(f"Error calculating or logging metrics: {e}")
            # Potentially log the error or handle it

        # --- Generate and Log Confusion Matrix ---
        try:
            cm_title = f'Confusion matrix ({var1})'
            # Construct filename based on how plot_confusion_matrix seems to save it (incorporating title)
            cm_filename = f'{cm_title}_{var1}.png' # Use the title in the filename
            cm_path = os.path.join(path_plot, cm_filename)

            # Assuming CLASS_NAMES is defined globally or passed
            cm_test1 = confusion_matrix(y_test_eval_int, predicted_eval_int)
            plot_confusion_matrix(path_plot, var1, cm_test1, classes=CLASS_NAMES, title=cm_title) # Util function saves the plot

            # Log the artifact using the corrected path
            mlflow.log_artifact(cm_path)
            print(f"Successfully logged confusion matrix: {cm_path}")

        except Exception as e:
            print(f"Error during confusion matrix generation/logging process: {e}")


        # --- Generate and Log Validation Curve (using evaluation data subset) ---
        try:
            title_val_curve_eval = f'validation_curve_eval_{var1}.png' # Use a different name
            palette1 = ["#8AA6A3","#127369"]
            X_train_selected = X_train1[:, input_best_index1] # Use selected features from training data

            curva_validacion3(fbest_model1, X_train_selected, y_train1, title_val_curve_eval, palette1, var1)
            plt.grid()
            fig = plt.gcf()
            val_curve_eval_path = os.path.join(path_plot, title_val_curve_eval)
            fig.savefig(val_curve_eval_path, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact(val_curve_eval_path)
        except Exception as e:
            print(f"Error generating/logging validation curve during evaluation: {e}")


        # --- Log Model Artifacts ---
        try:
            # Log the model using joblib
            model_path_joblib = os.path.join(path_plot, "model.joblib")
            joblib.dump(fbest_model1, model_path_joblib)
            mlflow.log_artifact(model_path_joblib)

            # Log the model using MLflow's built-in sklearn flavor
            # Create input example from training data (using selected features)
            # FIXME: Adjust the input example - using first training sample with selected features
            if X_train1.shape[0] > 0:
                 X_train_selected_sample = X_train1[0:1, input_best_index1]
                 input_example = pd.DataFrame(X_train_selected_sample, columns=[CLASS_NAMES[i] for i in input_best_index1] if len(input_best_index1) == len(CLASS_NAMES) else [f"feature_{i}" for i in input_best_index1]) # Create df for schema
                 print("Input example for MLflow model logging:", input_example)
            else:
                 input_example = None
                 print("Warning: Training data is empty, cannot create input example for MLflow model.")

            # Define registered model name
            registered_model_name = f"{NEURO}_{NAME}_{SPACE}_{STR_RATIO}"

            mlflow.sklearn.log_model(
                sk_model=fbest_model1,
                artifact_path="model", # Log under 'model' directory within run artifacts
                input_example=input_example,
                registered_model_name=registered_model_name
            )
            print(f"Model logged to MLflow registry as: {registered_model_name}")

        except Exception as e:
            print(f"Error logging model artifacts: {e}")


        return fbest_model1 # Return the evaluated model


# --- Main Execution Block ---
if __name__ == "__main__":
    # Define Constants/Parameters (should ideally come from config or CLI args)
    CLASS_NAMES = ['ACr', 'HC'] # Should match the keys in 'clases_mapeadas1' *after* mapping
    NEURO = 'neuroHarmonize'
    NAME = 'G1'
    SPACE = 'ic'
    # ICA = '58x25' # Not used directly in the script flow shown
    # RATIO = 79 # Not used directly in the script flow shown
    STR_RATIO = '2to1' # Used as 'var1'

    # Use absolute path for robustness, especially with Docker/Prefect
    PROJECT_PATH = os.path.abspath('/Users/imeag/Documents/udea/trabajoDeGrado/MLOps') # Make sure this path is correct or relative
    DATA_FILE_PATH = os.path.join(PROJECT_PATH, 'data', 'processed', f'Data_integration_ic_{NEURO}_{NAME}.feather')
    PATH_SAVE_BASE = os.path.join(PROJECT_PATH, 'Experimenting', 'Results') # Base path for saving results
    PATH_SAVE_RUN = os.path.join(PATH_SAVE_BASE, TIME_NOW) # Specific run path


    print(f"Starting script execution at: {TIME_NOW}")
    print(f"Data path: {DATA_FILE_PATH}")
    print(f"Results path: {PATH_SAVE_RUN}")

    # Define and set the experiment *before* the main run starts
    experiment_name = f"{NEURO}_{NAME}_{SPACE}_{STR_RATIO}"
    mlflow.set_experiment(experiment_name)
    print(f"MLflow experiment set to: {experiment_name}")


    # Start the main MLflow run context
    with mlflow.start_run(run_name=f"main_pipeline_{TIME_NOW}") as main_run:
        print(f"Main MLflow Run ID: {main_run.info.run_id}")
        # Log parameters for the main run
        mlflow.log_param("experiment_name", experiment_name)
        mlflow.log_param("project_path", PROJECT_PATH)
        mlflow.log_param("data_file_path", DATA_FILE_PATH)
        mlflow.log_param("neuro_param", NEURO)
        mlflow.log_param("name_param", NAME)
        mlflow.log_param("space_param", SPACE)
        mlflow.log_param("ratio_param", STR_RATIO)

        # --- Execute Training Phase (exec1 logic) ---
        # Pass data_path instead of loaded data
        training_results = exec1(
            NEURO, NAME, SPACE, PATH_SAVE_RUN, DATA_FILE_PATH, STR_RATIO, CLASS_NAMES, model=None
        )

        # Unpack results carefully, checking for None if exec1 could fail
        if training_results:
            (acc1, std1, fbest_model1, input_best_index1,
            X_train1, y_train1, clases_mapeadas1, var1,
            predicted_proba, X_test1, y_test1, path_plot_from_exec1) = training_results

            # --- Execute Evaluation Phase (exec2 logic) ---
            # Ensure required inputs are available before calling exec2
            if fbest_model1 is not None and input_best_index1 is not None:
                 # Use the path_plot generated and created within exec1
                 print(f"Using plot path for evaluation: {path_plot_from_exec1}")

                 final_model = exec2(
                     acc1, std1, fbest_model1, input_best_index1,
                     X_train1, y_train1, clases_mapeadas1, path_plot_from_exec1, var1, # Pass the correct path from exec1
                     predicted_proba, X_test1, y_test1
                 )

                 if final_model:
                     print("Script finished successfully. Final model evaluated.")
                 else:
                     print("Evaluation phase (exec2) failed.")
            else:
                 print("Training phase did not produce a valid model or feature indices. Skipping evaluation.")
        else:
            print("Training phase (exec1) failed or returned None.")

    print(f"Script execution finished.")
# --- Original Global Variables (potentially move into main or config) ---
# fbest_model1 = None # State should be managed within the functions/flow


# --- Old exec1 structure (for reference before removal) ---
# def exec1_original(...): ... # Keep temporarily if needed for comparison


# --- Old exec2 structure (for reference before removal) ---
# def exec2_original(...): ... # Keep temporarily if needed for comparison
