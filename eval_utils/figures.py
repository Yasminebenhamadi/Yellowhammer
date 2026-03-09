import os
import math
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix

# Plot functions
def plot_score_hist(eval_plot_folder, score_list, distance_list, model_name, show):
    df_results = pd.DataFrame(np.array([score_list, distance_list]).T, columns=['score', 'distance'])
    
    negatives_scores = []

    fig, axs = plt.subplots(1,7, figsize=(25, 3))

    i = 0
    for distance, frame in df_results.groupby(['distance']):
        if not math.isnan(distance[0]):
            negatives_scores = frame['score']
        else:
            distance_scores = frame['score']
            axs[i].hist(distance_scores, density=True, label="Positive", alpha=0.5)
            axs[i].hist(negatives_scores, density=True, label="Negative", alpha=0.5)
            axs[i].set_title("Scores at "+str(distance[0]), y=-0.2)
            
            i = i + 1

    plt.legend()
    plt.suptitle('Histogram of (predicted) scores of '+ model_name)
    
    if show:
        plt.show()
    else:
        folder_save = eval_plot_folder+"histograms/"
        os.makedirs(folder_save, exist_ok=True)
        plt.savefig(folder_save+model_name+'.png')
        plt.clf()

def plot_conf_matrix(eval_plot_folder, model_name, y_test=None, y_pred=None, conf_matrix=None):
    plt.figure(figsize=(6.4, 4.8))
    if conf_matrix == None:
        conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix "+model_name)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    folder_save = eval_plot_folder+"conf_matrices/"
    os.makedirs(folder_save, exist_ok=True)
    plt.savefig(folder_save+model_name+'.png')
    plt.clf()

def plot_curve_at_distance(eval_plot_folder, list_all, names, plot_name, show=False):
    plt.figure(figsize=(6.4, 4.8))
    for r in range(len(list_all)):
        plt.plot(list_all[r], '-o', label=names[r], linestyle="dotted")
    distances_unique = np.array([6.5, 12.5, 25, 50, 100, 150, 200])
    plt.legend()
    plt.grid(linestyle='--',alpha=0.3)
    plt.xticks(range(distances_unique.shape[0]),distances_unique)
    plt.yticks(np.linspace(start=0, stop=1, num=11))
    plt.xlabel("Distance (m)")
    plt.ylabel(plot_name)    
    if show:
        plt.show()
    else:
        plt.savefig(eval_plot_folder+plot_name+'.png')
        plt.clf()

def plot_ordered(eval_plot_folder, list_plot, names, title, color, ylabel="", show=False):
    list_plot = np.array(list_plot).flatten()
    names = np.array(names)
    sort_indx = list_plot.argsort()

    plt.figure(figsize=(15, 10), dpi=80)

    list_plot_sorted = np.round(list_plot[sort_indx], 2)

    plt.bar(range(len(list_plot)), list_plot_sorted, width=0.2, color=color)
    for i in range(len(list_plot)):
        plt.text(i,list_plot_sorted[i]*1.01,list_plot_sorted[i], ha = 'center')
    plt.xticks(range(len(list_plot)), names[sort_indx])
    plt.title(title)
    plt.xlabel("Models")
    plt.ylabel(ylabel)
    if show:
        plt.show()
    else:
        plt.savefig(eval_plot_folder+title+'.png')
        plt.clf()



def plot_conf(model, X, y_true):
    y_one = pd.get_dummies(y_true).astype(int).to_numpy()

    y_pred_scores = model.predict(X)
    pred_y_main = np.zeros_like(y_pred_scores)
    pred_y_main[np.arange(len(y_pred_scores)), np.argmax(y_pred_scores, axis=1)] = 1

    def plot_conf_matrix_multi(conf_matrix):
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

    conf_matrix = multilabel_confusion_matrix(y_one, pred_y_main)
    for i in range(len(conf_matrix)):
        print(np.sum(conf_matrix[i], axis=1))
        plot_conf_matrix_multi(conf_matrix[i])
        tn, fp, fn, tp = conf_matrix[i].ravel().tolist()
        recall=tp/(tp+fn+1e-9)
        precision=tp/(tp+fp+1e-9)
        print(recall, precision)