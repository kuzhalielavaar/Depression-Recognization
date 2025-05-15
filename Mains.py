import numpy as np
import pandas as pd
from numpy import matlib
import mne
import os
from keras.src.utils import to_categorical
from DPO import DPO
from DSO import DSO
from EPC import EPC
from GTO import GTO
from Global_Vars import Global_Vars
from Model_CNN import Model_CNN
from Model_MFF_Ada_ResCapsnet import Model_MFF_Ada_ResCapsnet
from Model_Descion_Tree import Model_Descion_Tree
from Model_STA_SAe import Model_STA_SAe
from Model_SVM import Model_SVM
from Objective_Function import objfun_cls
from PROPOSED import PROPOSED
from Plot_Results import *

no_of_dataset = 2


def Read_Dataset(Directory):
    Data = []
    Target = []
    listFiles = os.listdir(Directory)
    for i in range(len(listFiles)):
        filename = Directory + listFiles[i]
        data = np.load(filename, allow_pickle=True)
        for j in range(len(data)):
            for k in range(data[j].shape[0]):
                print(i, j, k)
                subdata = data[j][k, :, :]
                Data.append(subdata[:14, :].reshape(-1))
                Target.append(0)
                Data.append(subdata[14:20, :].reshape(-1))
                Target.append(1)
                Data.append(subdata[20:29, :].reshape(-1))
                Target.append(2)
                Data.append(subdata[29:, :].reshape(-1))
                Target.append(3)
    Min = np.min([len(i) for i in Data])
    Data = [i[:Min] for i in Data]
    Target = np.asarray(Target).reshape(-1, 1)
    cls_tar = (to_categorical(Target)).astype('int')
    return np.asarray(Data), cls_tar


# Read Dataset 1
an = 0
if an == 1:
    EEG_data, EEG_tar = Read_Dataset('./Datasets/Dataset_1/')
    np.save('Data_1.npy', EEG_data)
    np.save('Target_1.npy', EEG_tar)

# Read the Dataset 2
an = 0
if an == 1:
    Datasets = './Datasets/Dataset_2/'
    Dataset_path = os.listdir(Datasets)
    Datas = []
    Target = []
    for n in range(len(Dataset_path)):
        name = Dataset_path[n].split('.')[0]
        if name[0] == 'S' and name[-2:] == 'EC':
            if Dataset_path[n] != 'S8EC.edf':
                Data_dir_EC = Datasets + Dataset_path[n]
                Data_dir_EO = Datasets + name[:-2] + 'EO.edf'
                raw_EC = mne.io.read_raw_edf(Data_dir_EC, preload=True)
                raw_EO = mne.io.read_raw_edf(Data_dir_EO, preload=True)
                EC_data = raw_EC.get_data()
                EO_data = raw_EO.get_data()
                channel_names = raw_EC.ch_names

                for c in range(len(EC_data)):
                    Datas.append(EC_data[c])
                    Target.append(0)
                for o in range(len(EO_data)):
                    Datas.append(EO_data[o])
                    Target.append(0)

        elif name[0] == 'M' and name[-2:] == 'EC':
            Data_dir_EC = Datasets + Dataset_path[n]
            Data_dir_EO = Datasets + name[:-2] + 'EO.edf'
            raw_EC = mne.io.read_raw_edf(Data_dir_EC, preload=True)
            raw_EO = mne.io.read_raw_edf(Data_dir_EO, preload=True)
            EC_data = raw_EC.get_data()
            EO_data = raw_EO.get_data()
            channel_names = raw_EC.ch_names

            for c in range(len(EC_data) - 1):
                Datas.append(EC_data[c])
                Target.append(1)
            for o in range(len(EO_data) - 1):
                Datas.append(EO_data[o])
                Target.append(1)

    Min = np.min([len(i) for i in Datas])
    EEG_data = [i[:Min] for i in Datas]
    Target = np.asarray(Target).reshape(-1, 1)

    index = np.arange(len(EEG_data))
    np.random.shuffle(index)
    EEG_data = np.asarray(EEG_data)
    Shuffled_EEG_data = EEG_data[index]
    Shuffled_Target = Target[index]

    np.save('Index_2.npy', index)
    np.save('Data_2.npy', Shuffled_EEG_data)
    np.save('Target_2.npy', Shuffled_Target)


# Spatial Temporal Attention based Sparse Autoencoder Feature
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Positive_signal = np.load('Positive_signal_' + str(n + 1) + '.npy', allow_pickle=True)
        Neutral_signal = np.load('Neutral_signal_' + str(n + 1) + '.npy', allow_pickle=True)
        Negative_signal = np.load('Negative_signal_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        Feature_Pos = Model_STA_SAe(Positive_signal, Target)
        Feature_Neu = Model_STA_SAe(Neutral_signal, Target)
        Feature_Neg = Model_STA_SAe(Negative_signal, Target)
        np.save('Feature_Pos_' + str(n + 1) + '.npy', Feature_Pos)
        np.save('Feature_Neu_' + str(n + 1) + '.npy', Feature_Neu)
        np.save('Feature_Neg_' + str(n + 1) + '.npy', Feature_Neg)

# Optimization for Classification
an = 0
if an == 1:
    BEST_SOl = []
    FITNESS = []
    for n in range(no_of_dataset):
        Feat_1 = np.load('Feature_Pos_' + str(n + 1) + '.npy', allow_pickle=True)
        Feat_2 = np.load('Feature_Neg_' + str(n + 1) + '.npy', allow_pickle=True)
        Feat_3 = np.load('Feature_Neu_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        Global_Vars.Feat_1 = Feat_1
        Global_Vars.Feat_2 = Feat_2
        Global_Vars.Feat_3 = Feat_3
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 3  # Hidden Neuron Count, epoch, Learning rate in MFF_Ada_ResCapsnet
        xmin = matlib.repmat(np.asarray([5, 5, 0.01]), Npop, 1)
        xmax = matlib.repmat(np.asarray([255, 50, 0.99]), Npop, 1)
        fname = objfun_cls
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 50

        print("GTO...")
        [bestfit1, fitness1, bestsol1, time1] = GTO(initsol, fname, xmin, xmax, Max_iter)  # EVO

        print("EPC...")
        [bestfit2, fitness2, bestsol2, time2] = EPC(initsol, fname, xmin, xmax, Max_iter)  # GRO

        print("DPO...")
        [bestfit3, fitness3, bestsol3, time3] = DPO(initsol, fname, xmin, xmax, Max_iter)  # CLO

        print("DSO...")
        [bestfit4, fitness4, bestsol4, time4] = DSO(initsol, fname, xmin, xmax, Max_iter)  # DSO

        print("PROPOSED...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # PROPOSED

        BestSol_CLS = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(),
                       bestsol5.squeeze()]
        fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]

        BEST_SOl.append(BestSol_CLS)
        FITNESS.append(fitness)
    np.save('Fittness.npy', np.asarray(FITNESS))
    np.save('BestSol_CLS.npy', np.asarray(BEST_SOl))

# Classification
an = 0
if an == 1:
    EVAL_ALL = []
    for n in range(no_of_dataset):
        Feat_1 = np.load('Feature_Pos_' + str(n + 1) + '.npy', allow_pickle=True)
        Feat_2 = np.load('Feature_Neg_' + str(n + 1) + '.npy', allow_pickle=True)
        Feat_3 = np.load('Feature_Neu_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        BestSol = np.load('BestSol_CLS.npy', allow_pickle=True)[n]
        Feat = np.concatenate((Feat_1, Feat_2, Feat_3), axis=1)
        EVAL = []
        Batch_Size = [4, 16, 32, 64, 128]
        for BS in range(len(Batch_Size)):
            learnperc = round(Feat.shape[0] * 0.75)
            Train_Data = Feat[:learnperc, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Feat[learnperc:, :]
            Test_Target = Target[learnperc:, :]
            Eval = np.zeros((10, 25))
            for j in range(BestSol.shape[0]):
                print(BS, j)
                sol = np.round(BestSol[j, :]).astype(np.int16)
                Eval[j, :], pred = Model_MFF_Ada_ResCapsnet(Feat_1, Feat_2, Feat_3, Target, sol=sol, BS=Batch_Size[BS])
            Eval[5, :], pred1 = Model_SVM(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batch_Size[BS])
            Eval[6, :], pred2 = Model_Descion_Tree(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batch_Size[BS])
            Eval[7, :], pred3 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batch_Size[BS])
            Eval[8, :], pred4 = Model_MFF_Ada_ResCapsnet(Feat_1, Feat_2, Feat_3, Target, BS=Batch_Size[BS])
            Eval[9, :] = Eval[4, :]
            EVAL.append(Eval)
        EVAL_ALL.append(EVAL)
    np.save('Eval_all_BS.npy', np.asarray(EVAL_ALL))


plot_convergence()
ROC_curve()
Plot_Confusion()
Plot_batchsize()
Plot_Kfold()
Sample_images()
