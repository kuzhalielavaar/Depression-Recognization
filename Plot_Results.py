import numpy as np
from matplotlib import pylab
from sklearn.metrics import roc_curve
from itertools import cycle
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix

no_of_dataset = 2


def Statastical(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_convergence():
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'GTO-MFF-Ada-RCN', 'EPC-MFF-Ada-RCN', 'DPO-MFF-Ada-RCN ',
                 'DSOA-MFF-Ada-RCN', 'ERF-DSOA-MFF-Ada-RCN']
    Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
    Conv = np.zeros((Fitness.shape[-2], 5))
    for n in range(Fitness.shape[0]):
        for j in range(len(Algorithm) - 1):
            Conv[j, :] = Statastical(Fitness[n, j, :])
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv[j, :])
        print('-------------------------------------------------- Statistical Report of Dataset ', str(n + 1),
              '  --------------------------------------------------')
        print(Table)

        length = np.arange(Fitness.shape[-1])
        Conv_Graph = Fitness[n]
        plt.plot(length, Conv_Graph[0, :], color='#e50000', linewidth=3, label='GTO-MFF-Ada-RCN')
        plt.plot(length, Conv_Graph[1, :], color='#0504aa', linewidth=3, label='EPC-MFF-Ada-RCN')
        plt.plot(length, Conv_Graph[2, :], color='#0cff0c', linewidth=3, label='DPO-MFF-Ada-RCN')
        plt.plot(length, Conv_Graph[3, :], color='#9a0eea', linewidth=3, label='DSOA-MFF-Ada-RCN')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, label='ERF-DSOA-MFF-Ada-RCN')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Convergence Curve')
        plt.savefig("./Results/Convergence_Dataset_%s.png" % (n + 1))
        plt.show()


def ROC_curve():
    lw = 2

    Algorithm = ['TERMS', 'GTO-MFF-Ada-RCN', 'EPC-MFF-Ada-RCN', 'DPO-MFF-Ada-RCN ',
                 'DSOA-MFF-Ada-RCN', 'ERF-DSOA-MFF-Ada-RCN']
    cls = ['SVM', 'Decision Tree', 'CNN', 'MFF-Ada-RCN', 'ERF-DSOA-MFF\n-Ada-RCN']
    for n in range(no_of_dataset):

        Actual = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True).astype('int')
        colors = cycle(
            ["#fe2f4a", "#0165fc", "#f97306", "lime", "black"])
        for i, color in zip(range(len(cls)), colors):
            Predicted = np.load('Y_Score_' + str(n + 1) + '.npy', allow_pickle=True)[i]
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[i],
            )
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path = "./Results/ROC_Dataset_%s.png" % (n + 1)
        plt.savefig(path)
        plt.show()


def Plot_Confusion():
    for n in range(no_of_dataset):
        Actual = np.load('Actual_' + str(n + 1) + '.npy', allow_pickle=True)
        Predict = np.load('Predict_' + str(n + 1) + '.npy', allow_pickle=True)

        if n == 0:
            cm = confusion_matrix(np.asarray(Actual).argmax(axis=1), np.asarray(Predict).argmax(axis=1))
            Classes = ['minimal', 'mild', 'moderate', 'severe']
        else:
            cm = confusion_matrix(np.asarray(Actual), np.asarray(Predict))
            Classes = ['Normal', 'MDD']
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
        # cm_display.plot()
        fig, ax = plt.subplots(figsize=(8, 6))
        cm_display.plot(ax=ax, cmap='cividis',
                        values_format='d')  # Blues 'viridis', 'plasma', 'inferno', 'magma', 'cividis'

        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('Actual labels')
        ax.set_title('Confusion Matrix')

        ax.xaxis.set_ticklabels(Classes)
        ax.yaxis.set_ticklabels(Classes)

        path = "./Results/Confusion_Dataset_%s.png" % (n + 1)
        plt.title('Confusion matrix')
        plt.savefig(path)
        plt.show()


def Plot_batchsize():
    eval = np.load('Eval_ALL_BS.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 score', 'MCC',
             'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'lrplus', 'lrminus', 'DOR', 'prevalence']
    Graph_Terms = [0, 3, 8, 9, 10]
    for n in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = eval[n, :, :, Graph_Terms[j] + 4]
            Graph = Graph[:, :]
            X = np.arange(Graph.shape[0])
            ax = plt.axes()
            ax.set_facecolor("#BAFCFF")

            plt.plot(X, Graph[0, :5], color='#ff000d', linewidth=4, marker='$\spadesuit$', markerfacecolor='#ffff81',
                     markersize=12,
                     label="Batch size 4")
            plt.plot(X, Graph[1, :5], color='#0cff0c', linewidth=4, marker='$\diamondsuit$', markerfacecolor='red',
                     markersize=12,
                     label="Batch size 16")
            plt.plot(X, Graph[2, :5], color='#0652ff', linewidth=4, marker='$\clubsuit$', markerfacecolor='#bdf6fe',
                     markersize=12,
                     label="Batch size 32")
            plt.plot(X, Graph[3, :5], color='#FFAE00', linewidth=4, marker='$\U0001F601$', markerfacecolor='yellow',
                     markersize=12,
                     label="Batch size 64")
            plt.plot(X, Graph[4, :5], color='black', linewidth=4, marker='$\U00002660$', markerfacecolor='cyan',
                     markersize=12,
                     label="Batch size 128")

            plt.xticks(X, ('GTO-MFF-\nAda-RCN', 'EPC-MFF-\nAda-RCN', 'DPO-MFF-\nAda-RCN ',
                                    'DSOA-MFF-\nAda-RCN', 'ERF-DSOA-\nMFF-Ada-RCN'))
            plt.grid(axis='y', linestyle='--', color='gray', which='major', alpha=0.8)
            plt.ylabel(Terms[Graph_Terms[j]], fontsize=12, fontweight='bold', color='#35530a')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
            path = "./Results/Batch_size_%s_line_Dataset_%s.png" % (Terms[Graph_Terms[j]], n + 1)
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('Batch size vs ' + Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.12, 0.1, 0.8, 0.8])
            X = np.arange(Graph.shape[0])
            ax.set_facecolor("#BAFCFF")

            ax.bar(X + 0.00, Graph[0, 5:], color='#FFB830', width=0.15, label="Batch size 4")
            ax.bar(X + 0.15, Graph[1, 5:], color='#FF6600', width=0.15, label="Batch size 16")
            ax.bar(X + 0.30, Graph[2, 5:], color='#25EB03', width=0.15, label="Batch size 32")
            ax.bar(X + 0.45, Graph[3, 5:], color='#EB61D4', width=0.15, label="Batch size 64")
            ax.bar(X + 0.60, Graph[4, 5:], color='k', width=0.15, label="Batch size 128")

            plt.xticks(X + 0.3, ('SVM', 'Decision Tree', 'CNN', 'MFF-\nAda-RCN', 'ERF-DSOA-MFF\n-Ada-RCN'))
            plt.ylabel(Terms[Graph_Terms[j]], fontsize=12, fontweight='bold', color='#35530a')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
            plt.grid(axis='y', linestyle='--', color='gray', which='major', alpha=0.8)
            path = "./Results/Batch_size_%s_bar_Dataset_%s.png" % (Terms[Graph_Terms[j]], n + 1)
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('Batch size vs ' + Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def Plot_Kfold():
    eval = np.load('Eval_ALL_Fold.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'lrplus', 'lrminus', 'DOR', 'prevalence']
    Table_Term = [0, 2, 4, 7, 8, 9, 10, 12]
    k_fold = ['1', '2', '3', '4', '5']

    Algorithm = ['TERMS', 'GTO-MFF-Ada-RCN', 'EPC-MFF-Ada-RCN', 'DPO-MFF-Ada-RCN ',
                 'DSOA-MFF-Ada-RCN', 'ERF-DSOA-MFF-Ada-RCN']
    Classifier = ['TERMS', 'SVM', 'Decision Tree', 'CNN', 'MFF-Ada-RCN', 'ERF-DSOA-MFF\n-Ada-RCN']
    for n in range(eval.shape[0]):
        for k in range(eval.shape[1]):
            value = eval[n, k, :, 4:]
            Table = PrettyTable()
            Table.add_column(Algorithm[0], (np.asarray(Terms))[np.asarray(Table_Term)])
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[j, Table_Term])
            print('-------------------------------------------------- ', str(k_fold[k]), ' Fold of dataset', str(n + 1),
                  'Algorithm Comparison --------------------------------------------------')
            print(Table)
            Table = PrettyTable()
            Table.add_column(Classifier[0], (np.asarray(Terms))[np.asarray(Table_Term)])
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, Table_Term])
            print('-------------------------------------------------- ', str(k_fold[k]), ' Fold of dataset', str(n + 1),
                  'Classifier Comparison --------------------------------------------------')
            print(Table)


def Sample_images():
    for n in range(no_of_dataset):
        Images = np.load('Data_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        if Target.shape[-1] > 1:
            targ = np.argmax(Target, axis=1).reshape(-1, 1)
        else:
            targ = Target
        class_indices = {}
        for class_label in np.unique(targ):
            indices = np.where(targ == class_label)[0]
            class_indices[class_label] = indices
        for class_label, indices in class_indices.items():
            if n == 0:
                labels = ['minimal', 'mild', 'moderate', 'severe']
            if n == 1:
                labels = ['Normal', 'MDD']
            for i in range(5, 10):
                print(labels[class_label], i - 4)
                Image = Images[indices[-i]]
                plt.plot(Image)
                plt.xlabel('Time (ms)')
                plt.ylabel('Amplitude')
                plt.tight_layout()
                path = './Results/Sample_Images/Dataset_' + str(n + 1) + '_' + str(
                    labels[class_label]) + '_image_' + str(
                    i - 4) + '.png'
                plt.savefig(path)
                plt.show()


if __name__ == '__main__':
    plot_convergence()
    ROC_curve()
    Plot_Confusion()
    Plot_batchsize()
    Plot_Kfold()
    Sample_images()
