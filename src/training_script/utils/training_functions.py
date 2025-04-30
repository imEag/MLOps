import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from tpot import TPOTClassifier
import seaborn as sns
from sklearn.model_selection import train_test_split, \
                                    cross_val_score, \
                                    learning_curve, \
                                    RandomizedSearchCV, \
                                    GridSearchCV
from sklearn.tree import export_graphviz
import pydot
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from collections import defaultdict
import csv

## Permite correr dos archivos pero esta pensado para dos flujos diferentes, como sovaharmony y neuroharmonaze
## Si busca realizar comparaciones utilice el archivo training_script_2.py 
## Se utiliza cuando se quieren comparar 2 archivos que tengan una variación, por ejemplo diferentes ICA, por eso contine los parametros Var1 y Var2

def compute_precision_recall(test_label, classes_x):
    precision_test = precision_score(test_label, classes_x)
    recall_test = recall_score(test_label, classes_x)
    f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)
    print('Precision: ', precision_test, '\n', 'Recall: ', recall_test, '\n', 'F1-score:', f1_test)

def plot_confusion_matrix(path_plot,var, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.BuGn):
    if var == '2to1':
        var2 = '2:1'
    elif var == '5to1':
        var2 = '5:1'
    elif var == '10to1':
        var2 = '10:1'
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(f"Confusion Matrix normalize {var2}")
    else:
        print(f'Confusion Matrix {var2}')

    print(cm)
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     fontsize=20,  # Tamaño de fuente ajustado a 20
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    plt.tight_layout()
    plt.savefig(path_plot+'/'+title+'_'+var+'.png')
    plt.close()

def delcol(data, m, b, roi, bm=None):
    for metrici in m:
        for bandi in b:
            for ri in roi:
                if metrici != 'crossfreq':
                    data.drop([metrici+'_'+ri+'_'+bandi], axis=1, inplace=True)
                else:
                    for modul in bm:
                        if modul == 'MGAMMA':
                            modul = modul[0] + modul[1:].swapcase()
                        else:
                            pass
                        try:
                            data.drop([metrici+'_'+ri+'_'+modul+'_'+bandi], axis=1, inplace=True)
                        except:
                            continue
    return data


def mapa_de_correlacion(data, path_plot,var):
    # Calcular la matriz de correlación de Pearson
    correlation_matrix = data.corr()
    # Generar la gráfica de correlación antes de eliminar características
    plt.figure(figsize=(15, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, annot_kws={"size": 5}, cbar=True)
    plt.title(f"Correlation Matrix for {var} Ratio",fontsize=20)
    plt.tight_layout()
    # Ensure the directory exists before saving
    if not os.path.exists(path_plot):
        os.makedirs(path_plot, exist_ok=True)
    plt.savefig(os.path.join(path_plot, 'correlation_before.png'))
    plt.close()

    # Configurar el umbral de correlación
    threshold = 0.8

    # Listas para almacenar características altamente correlacionadas y eliminadas
    highly_correlated_features = []
    features_to_drop = []

    # Encontrar pares de características con correlación mayor al umbral
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname_i = correlation_matrix.columns[i]
                colname_j = correlation_matrix.columns[j]
                highly_correlated_features.append((colname_i, colname_j))
                if colname_i not in features_to_drop:
                    features_to_drop.append(colname_i)

    # Eliminar las características altamente correlacionadas
    data_reduced = data.drop(columns=features_to_drop)

    # Calcular la nueva matriz de correlación
    new_correlation_matrix = data_reduced.corr()

    # Generar la gráfica de correlación después de eliminar características
    plt.figure(figsize=(15, 10))
    sns.heatmap(new_correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title(f"Reduced Correlation Matrix for {var} Ratio",fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(path_plot, 'correlation_after.png'))
    plt.close()

    # Guardar el DataFrame reducido como archivo Feather
    feather_file_path = os.path.join(path_plot, f'Data_integration_corr.feather')
    data_reduced.reset_index(drop=True).to_feather(feather_file_path)
    return data_reduced

def grid_search():
    # Número de árboles en el bosque
    n_estimators = [100, 200, 300] #Antes de ChatGPT [int(x) for x in np.linspace(start = 100, stop = 2000, num = 30)]

    # Número máximo de características a considerar en cada división
    #max_features = ['auto', 'sqrt', 'log2']
    max_features = ['sqrt', 'log2'] # Cambio 27/05/2024

    # Profundidad máxima del árbol
    max_depth = [5, 10, 15, None] #Antes de ChatGPT [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)

    # Número mínimo de muestras requeridas para dividir un nodo
    min_samples_split = [10, 20, 30]

    # Número mínimo de muestras requeridas en cada hoja
    min_samples_leaf = [5, 10, 15]

    # Si se deben realizar remuestreos con reemplazo
    bootstrap = [True, False]

    # Criterio para medir la calidad de una división
    criterion = ['gini', 'entropy']

    # Crear el conjunto de hiperparámetros
    random_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap,
        'criterion': criterion
    }
    return random_grid

def randomFo(random_grid,X_train, y_train):
    forestclf_grid = RandomForestClassifier()
    #n_iter=100, cv=2
    rf_random = RandomizedSearchCV(
                                estimator=forestclf_grid,
                                param_distributions=random_grid,
                                n_iter=100, 
                                cv=10,
                                verbose=2,
                                random_state=10,
                                n_jobs=-1
                                )

    rf_random.fit(X_train, y_train)
    return rf_random

def curva_validacion(GS_fitted,X_train,y_train,path_plot,title):
    train_sizes, train_scores, test_scores = \
    learning_curve(
                    estimator=GS_fitted,
                    X=X_train,
                    y=y_train,
                    train_sizes=np.linspace(0.1, 0.8, 8),
                    cv=10,
                    n_jobs=-1
                    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(
            train_sizes,
            train_mean,
            color='blue',
            marker='o',
            markersize=5,
            label='training accuracy'
            )

    plt.fill_between(
                    train_sizes,
                    train_mean + train_std,
                    train_mean - train_std,
                    alpha=0.15,
                    color='blue'
                    )

    plt.plot(
            train_sizes,
            test_mean,
            color='green',
            linestyle='--',
            marker='s',
            markersize=5,
            label='validation accuracy'
            )

    plt.fill_between(
                    train_sizes,
                    test_mean + test_std,
                    test_mean - test_std,
                    alpha=0.15,
                    color='green'
                    )

    plt.grid()
    plt.title('Validation curve for ' + title[11:-4])
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.5, 1.0])

    plt.savefig(path_plot+'/'+title)
    plt.close()

def curva_validacion3(GS_fitted, X_train, y_train, title, palette, var):
    train_sizes, train_scores, test_scores = \
    learning_curve(
                    estimator=GS_fitted,
                    X=X_train,
                    y=y_train,
                    train_sizes=np.linspace(0.1, 0.8, 8),
                    cv=10,
                    n_jobs=-1
                    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(
            train_sizes,
            train_mean,
            color=palette[0],
            marker='o',
            markersize=5,
            label='training accuracy '
            )

    plt.fill_between(
                    train_sizes,
                    train_mean + train_std,
                    train_mean - train_std,
                    alpha=0.15,
                    color=palette[0]
                    )

    plt.plot(
            train_sizes,
            test_mean,
            color=palette[1],
            linestyle='--',
            marker='s',
            markersize=5,
            label='validation accuracy '
            )

    plt.fill_between(
                    train_sizes,
                    test_mean + test_std,
                    test_mean - test_std,
                    alpha=0.15,
                    color=palette[1]
                    )

    
    if var == '2to1':
        var2 = '2:1'
    elif var == '5to1':
        var2 = '5:1'
    elif var == '10to1':
        var2 = '10:1'
    plt.grid()
    plt.title('Training and Validation Curves for ' + title[11:-4]+' '+var2, fontsize=20)
    plt.xlabel('Number of training samples', fontsize=20)
    plt.ylabel('Score', fontsize=20)

    # Modificar la posición de la leyenda
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=12)

    # Ajustar el límite visualmente a 1 sin cambiar el rango real hasta 1.1
    #plt.ylim([0.5, 1.2])
    plt.gca().set_ylim([0.3, 1.1])
    nuevos_valores_y = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    plt.yticks(nuevos_valores_y, fontsize=20)
    plt.xticks(fontsize=20)
    plt.grid()

def curva_validacion2(GS_fitted1,GS_fitted2,X_train1,y_train1,X_train2,y_train2,path_plot,title,cross):
    palette = ["#8AA6A3","#127369","#10403B","#45C4B0"]
    
    train_sizes1, train_scores1, test_scores1 = learning_curve(
        estimator=GS_fitted1,
        X=X_train1,
        y=y_train1,
        train_sizes=np.linspace(0.1, 0.8, 8),
        cv=cross,
        n_jobs=-1
    )
    train_mean1 = np.mean(train_scores1, axis=1)
    train_std1 = np.std(train_scores1, axis=1)
    test_mean1 = np.mean(test_scores1, axis=1)
    test_std1 = np.std(test_scores1, axis=1)

    train_sizes2, train_scores2, test_scores2 = learning_curve(
        estimator=GS_fitted2,
        X=X_train2,
        y=y_train2,
        train_sizes=np.linspace(0.1, 0.8, 8),
        cv=cross,
        n_jobs=-1
    )
    train_mean2 = np.mean(train_scores2, axis=1)
    train_std2 = np.std(train_scores2, axis=1)
    test_mean2 = np.mean(test_scores2, axis=1)
    test_std2 = np.std(test_scores2, axis=1)

    plt.fill_between(
                train_sizes1,
                train_mean1 + train_std1,
                train_mean1 - train_std1,
                alpha=0.15,
                color=palette[0]
                )
    plt.plot(
            train_sizes1,
            train_mean1,
            color=palette[0],
            marker='o',
            markersize=5,
            label='training accuracy 58x25'
            )
    
    plt.fill_between(
            train_sizes2,
            train_mean2 + train_std2,
            train_mean2 - train_std2,
            alpha=0.15,
            color=palette[1]
            )
    
    plt.plot(
            train_sizes2,
            train_mean2,
            color=palette[1],
            marker='o',
            markersize=5,
            label='training accuracy 54x10'
            )

    plt.fill_between(
                train_sizes1,
                test_mean1 + test_std1,
                test_mean1 - test_std1,
                alpha=0.15,
                color=palette[2]
                )
    plt.plot(
            train_sizes1,
            test_mean1,
            color=palette[2],
            linestyle='--',
            marker='s',
            markersize=5,
            label='validation accuracy 58x25'
            )
    
    plt.fill_between(
                train_sizes2,
                test_mean2 + test_std2,
                test_mean2 - test_std2,
                alpha=0.15,
                color=palette[3]
                )
    plt.plot(
            train_sizes2,
            test_mean2,
            color=palette[3],
            linestyle='--',
            marker='s',
            markersize=5,
            label='validation accuracy 54x10'
            )

    plt.grid()
    plt.title('Validation curve for ' + title[11:-4])
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')

    # Modificar la posición de la leyenda
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.85), ncol=2)

    # Ajustar el límite visualmente a 1 sin cambiar el rango real hasta 1.1
    #plt.ylim([0.5, 1.2])
    plt.gca().set_ylim([0.5, 1.1])
    nuevos_valores_y = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    plt.yticks(nuevos_valores_y)

    fig = plt.gcf()
    fig.savefig(path_plot + '/' + title)
    plt.close()


def primeras_carateristicas(X_train, sorted_names,nombres_columnas,features_scores,feat,index,path_plot,var):
    for f in range(X_train.shape[1]):
        sorted_names.append(nombres_columnas[index[f]])
        print("%2d) %-*s %f" % (f + 1, 30,
                            nombres_columnas[index[f]],
                            features_scores[index[f]]))
        feat[nombres_columnas[index[f]]] = [features_scores[index[f]]]
    plt.title(f'Analyzing Feature Importance {var}')
    plt.xlabel('Features')
    plt.ylabel('Relevance')

    plt.bar(range(X_train.shape[1]),
        features_scores[index],
        align='center')

    plt.xticks(range(X_train.shape[1]),
        nombres_columnas[index],
        rotation=90)

    plt.xlim([-1, 10])

    plt.tight_layout()
    plt.savefig(path_plot+'/'+'features_table_plot_all.png')
    plt.close()
    return feat

def grafico_arbol_de_decision(best_selected, nombres_columnas, output_file="decision_tree"):
    # Selecciona un árbol individual del RandomForest (por ejemplo, el primer árbol)
    individual_tree = best_selected.estimators_[2]

    # Genera el gráfico del árbol de decisión
    dot_data = export_graphviz(individual_tree, out_file=None,
                               feature_names=nombres_columnas,
                               class_names=[str(x) for x in best_selected.classes_],
                               filled=True, rounded=True,
                               special_characters=True)

    # Cambia el color de fondo de todos los nodos a verde (incluidas las hojas)
    dot_data = dot_data.replace('fillcolor="#e5813900"', 'fillcolor="green"')

    # Crea un objeto Dot
    graph = pydot.graph_from_dot_data(dot_data)

    # Guarda el gráfico en un archivo de imagen (formato PNG)
    graph[0].write_png(f"{output_file}.png")

    # Muestra el gráfico del árbol en una ventana emergente
    img = Image.open(f"{output_file}.png")
    img.show()

def curva_de_aprendizaje(sorted_names,data,best_selected,X_train,y_train,modelos,acc_per_feature,std_per_feature,path_plot,var):
    for index, feature_name in enumerate(sorted_names,start=1):

        input_features_names = sorted_names[:index]
        input_features_index = [data.columns.get_loc(c) for c
                        in input_features_names if c in data]
        feature_model = best_selected.fit(X_train[:, input_features_index], y_train)
        scores = cross_val_score(
                            estimator=feature_model,
                            X=X_train[:, input_features_index],
                            y=y_train,
                            cv=10,
                            n_jobs=-1
                            )


        modelos['number_features_' + str(index)] = feature_model
        acc_per_feature.append(np.mean(scores))
        std_per_feature.append(np.std(scores))


    #plt.plot(
    #        range(1, len(sorted_names)),
    #        acc_per_feature,
    #        color='red'
    #        )
    #
    #plt.fill_between(
    #                range(1, len(sorted_names)),
    #                np.array(acc_per_feature) + np.array(std_per_feature),
    #                np.array(acc_per_feature) - np.array(std_per_feature),
    #                alpha=0.15,
    #                color='red'
    #                )

    plt.plot(
            range(1, len(acc_per_feature)+1),
            acc_per_feature,
            color='red'
            )

    plt.fill_between(
                    range(1, len(acc_per_feature)+1),
                    np.array(acc_per_feature) + np.array(std_per_feature),
                    np.array(acc_per_feature) - np.array(std_per_feature),
                    alpha=0.15,
                    color='red'
                    )

    plt.grid()
    plt.title('Learning Curve Decision Tree'+' '+var)
    plt.xlabel('Number of features')
    plt.ylabel('Accuracy')
    plt.savefig(path_plot+'/'+'features_plot_all.png')
    plt.close()


def features_best(best_features,best_selected,data,X_train,y_train,path_plot):
    acc = []
    std = []
    m=[]

    # Verificar dimensiones de X_train
    print("Dimensiones de X_train:", X_train.shape)
    
    for index, feature_name in enumerate(best_features,start=1):

        input_features_best = best_features[:index]
        input_best_index = [data.columns.get_loc(c) for c
                        in input_features_best if c in data]
        fbest_model = best_selected.fit(X_train[:, input_best_index], y_train)
        scores_best = cross_val_score(
                            estimator=fbest_model,
                            X=X_train[:, input_best_index],
                            y=y_train,
                            cv=10,
                            n_jobs=-1
                            )


        #m['number_features_BEST' + str(index)] = fbest_model
        acc.append(np.mean(scores_best))
        std.append(np.std(scores_best))

    plt.plot(
            range(1, len(acc)+1),
            acc,
            color='red'
            )
    plt.title('Learning Curve Decision Tree')
    plt.xlabel('Number of features')
    plt.ylabel('Accuracy')

    plt.fill_between(
                    range(1, len(acc)+1),
                    np.array(acc) + np.array(std),
                    np.array(acc) - np.array(std),
                    alpha=0.15,
                    color='red'
                    )

    plt.grid()
    plt.savefig(path_plot+'/'+'features_plot_best.png')
    plt.close()
    return acc, std, fbest_model, input_best_index

def features_best3(best_features, best_selected, data, X_train, y_train, path_plot):
    acc = []
    std = []

    # Verificar dimensiones de X_train
    print("Dimensiones de X_train:", X_train.shape)

    for index, feature_name in enumerate(best_features, start=1):
        input_features_best = best_features[:index]
        input_best_index = [data.columns.get_loc(c) for c in input_features_best if c in data]
        
        # Verificar si el índice está fuera de rango
        if any(idx >= X_train.shape[1] for idx in input_best_index):
            print(f"Índice fuera de rango detectado en la iteración {index}: {input_best_index}")
            continue
        
        # Información de depuración
        print(f"Iteración {index}:")
        print("Características seleccionadas:", input_features_best)
        print("Índices de características:", input_best_index)

        try:
            fbest_model = best_selected.fit(X_train[:, input_best_index], y_train)
            scores_best = cross_val_score(
                estimator=fbest_model,
                X=X_train[:, input_best_index],
                y=y_train,
                cv=10,
                n_jobs=-1
            )

            acc.append(np.mean(scores_best))
            std.append(np.std(scores_best))
        except IndexError as e:
            print(f"Error de índice en la iteración {index}: {e}")
            continue

    if acc:  # Ensure acc is not empty to avoid errors in return statement
        plt.plot(range(1, len(acc) + 1), acc, color='red')
        plt.title('Learning Curve Decision Tree')
        plt.xlabel('Number of features')
        plt.ylabel('Accuracy')

        plt.fill_between(
            range(1, len(acc) + 1),
            np.array(acc) + np.array(std),
            np.array(acc) - np.array(std),
            alpha=0.15,
            color='red'
        )

        plt.grid()
        plt.savefig(path_plot + '/' + 'features_plot_best.png')
        plt.close()
    
    return acc, std, fbest_model if acc else None, input_best_index if acc else None

def features_best2(best_features1,best_features2,best_selected1,best_selected2,data1,data2,X_train1,y_train1,X_train2,y_train2,path_plot):
    acc1 = []
    std1 = []
    acc2 = []
    std2 = []
    m=[]
    for index, feature_name1 in enumerate(best_features1,start=1):

        input_features_best = best_features1[:index]
        input_best_index1 = [data1.columns.get_loc(c) for c
                        in input_features_best if c in data1]
        fbest_model1 = best_selected1.fit(X_train1[:, input_best_index1], y_train1)
        scores_best1 = cross_val_score(
                            estimator=fbest_model1,
                            X=X_train1[:, input_best_index1],
                            y=y_train1,
                            cv=10,
                            n_jobs=-1
                            )
    for index, feature_name2 in enumerate(best_features2,start=1):

        input_features_best = best_features2[:index]
        input_best_index2 = [data2.columns.get_loc(c) for c
                        in input_features_best if c in data2]
        fbest_model2 = best_selected2.fit(X_train2[:, input_best_index2], y_train2)
        scores_best2 = cross_val_score(
                            estimator=fbest_model2,
                            X=X_train2[:, input_best_index2],
                            y=y_train2,
                            cv=10,
                            n_jobs=-1
                            )

        acc1.append(np.mean(scores_best1))
        std1.append(np.std(scores_best1))
        
        acc2.append(np.mean(scores_best2))
        std2.append(np.std(scores_best2))

    plt.plot(
            range(1, len(acc1)+1),
            acc1,
            color='#EF3300'
            )
    plt.plot(
            range(1, len(acc2)+1),
            acc2,
            color='#F05E00'
            )
    plt.title('Learning Curve Decision Tree')
    plt.xlabel('Number of features')
    plt.ylabel('Accuracy')

    plt.fill_between(
                    range(1, len(acc1)+1),
                    np.array(acc1) + np.array(std1),
                    np.array(acc1) - np.array(std1),
                    alpha=0.15,
                    color='#EF3300'
                    )
    plt.fill_between(
                    range(1, len(acc2)+1),
                    np.array(acc2) + np.array(std2),
                    np.array(acc2) - np.array(std2),
                    alpha=0.15,
                    color='#F05E00'
                    )

    plt.grid()
    plt.savefig(path_plot+'/'+'features_plot_best.png')
    plt.close()
    return acc1, std1, fbest_model1, input_best_index1,acc2, std2, fbest_model2, input_best_index2

# compute precision and recall
def computerprecision(test_label, classes_x, output_file):
    precision_test = precision_score(test_label, classes_x)
    recall_test = recall_score(test_label, classes_x)
    f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)
    
    # Imprimir los resultados en la consola
    print('Precision: ', precision_test, '\n', 'Recall: ', recall_test, '\n', 'F1-score:', f1_test)
    
    # Guardar los resultados en un archivo
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metrica', 'Valor'])
        writer.writerow(['Precision', precision_test])
        writer.writerow(['Recall', recall_test])
        writer.writerow(['F1-score', f1_test])

def importancia():
    palette = ["#8AA6A3","#127369","#10403B","#45C4B0"]
    # Crear una instancia de Tkinter (necesaria para los cuadros de diálogo)
    root = Tk()
    root.withdraw()  # Ocultar la ventana principal de Tkinter

    # Abre un cuadro de diálogo para seleccionar un archivo
    file_path = askopenfilename(title="Seleccionar archivo", filetypes=[("Archivos de texto", "best_features.txt")])

    # Cargar el archivo seleccionado en un DataFrame
    if file_path:
        df = pd.read_csv(file_path, delimiter='\t', header=None)  # Ajustar el delimitador según sea necesario

        # Crear un DataFrame vacío
        categories_df = pd.DataFrame()

        # Extraer y contar las categorías usando listas de comprensión
        categories_df['Feature'] = [t.split('_')[0] for t in df[0]]
        categories_df['IC'] = [t.split('_')[1] if 'age' not in t else None for t in df[0]]
        #categories_df['ROI'] = [t.split('_')[1] if 'age' not in t and 'sex' not in t else None for t in df[0]]
        categories_df['Mband'] = [t.split('_')[2] if len(t.split('_')) == 4 else None for t in df[0]]
        categories_df['Band'] = [t.split('_')[2] if len(t.split('_')) >= 3 and 'age' not in t and 'sex' not in t and not t.split('_')[2].startswith('M') else t.split('_')[3] if 'age' not in t and 'sex' not in t else None for t in df[0]]

        # Crear subplots
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs = axs.flatten()
        fig.suptitle("Discriminant analysis of the most relevant features using Decition tree with neuroHarmonize", fontsize=15, x=0.55)  # 54x10
        # Graficar cada columna en un subplot
        for i, col in enumerate(categories_df.columns):
            ax = axs[i]
            if i == 0:  # Cambiar el color de la segunda barra del primer gráfico en (1, 1)
                ax.bar(categories_df[col].value_counts().index, categories_df[col].value_counts().values, color=[palette[0], palette[0], palette[1], palette[0], palette[0], palette[0]])
            elif i == 1:  
                ax.bar(categories_df[col].value_counts().index, categories_df[col].value_counts().values, color=[palette[0], palette[1], palette[0], palette[0], palette[0], palette[0], palette[0], palette[0], palette[0]])
            elif i == 2:  
                ax.bar(categories_df[col].value_counts().index, categories_df[col].value_counts().values, color=[palette[0], palette[0], palette[0], palette[1], palette[1], palette[0], palette[0], palette[0]])
            elif i == 3:  
                ax.bar(categories_df[col].value_counts().index, categories_df[col].value_counts().values, color=[palette[1], palette[0], palette[0], palette[0], palette[0], palette[0], palette[0], palette[1],palette[0], palette[0], palette[0], palette[0], palette[0], palette[0], palette[0], palette[0]])
            else:
                pass
            ax.set_title(col)
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)


        
        plt.tight_layout()