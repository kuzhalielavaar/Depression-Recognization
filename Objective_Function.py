import numpy as np
from Classificaltion_Evaluation import ClassificationEvaluation
from Global_Vars import Global_Vars
from Model_MFF_Ada_ResCapsnet import Model_MFF_Ada_ResCapsnet


def objfun_cls(Soln):
    Feat_1 = Global_Vars.Feat_1
    Feat_2 = Global_Vars.Feat_2
    Feat_3 = Global_Vars.Feat_3
    Feat = np.concatenate((Feat_1, Feat_2, Feat_3), axis=0)
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(Feat.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Test_Target = Tar[learnper:, :]
            Eval, pred = Model_MFF_Ada_ResCapsnet(Feat_1, Feat_2, Feat_3, Tar, sol=sol)
            Eval = ClassificationEvaluation(Test_Target, pred)
            Fitn[i] = (1 / Eval[13]) + Eval[11]  # (1 / MCC) + FDR
        return Fitn
    else:
        learnper = round(Feat.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Test_Target = Tar[learnper:, :]
        Eval, pred = Model_MFF_Ada_ResCapsnet(Feat_1, Feat_2, Feat_3, Tar, sol=sol)
        Eval = ClassificationEvaluation(Test_Target, pred)
        Fitn = (1 / Eval[13]) + Eval[11]  # (1 / MCC) + FDR
        return Fitn
