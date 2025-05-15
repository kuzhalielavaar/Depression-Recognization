from sklearn.svm import SVC  # "Support Vector Classifier"
import numpy as np
from Classificaltion_Evaluation import ClassificationEvaluation


def Model_SVM(train_data, train_target, test_data, test_target, Epoch=None, BS=None, sol=1):
    print('Model_SVM')
    if Epoch is None:
        Epoch = 5
    if BS is None:
        BS = 4
    Kernels = ['linear', 'poly', 'rbf', 'sigmoid']

    IMG_SIZE = 10
    Train_Temp = np.zeros((train_data.shape[0], IMG_SIZE))
    for i in range(train_data.shape[0]):
        Train_Temp[i, :] = np.resize(train_data[i], IMG_SIZE)
    train_data = Train_Temp.reshape(Train_Temp.shape[0], IMG_SIZE)

    Test_Temp = np.zeros((test_data.shape[0], IMG_SIZE))
    for i in range(test_data.shape[0]):
        Test_Temp[i, :] = np.resize(test_data[i], IMG_SIZE)
    test_data = Test_Temp.reshape(Test_Temp.shape[0], IMG_SIZE)

    if train_target.shape[-1] >= 2:
        train_tar = np.argmax(train_target, axis=1).reshape(-1)
        test_tar = np.argmax(test_target, axis=1).reshape(-1)
    else:
        train_tar = train_target.reshape(-1)
        test_tar = test_target.reshape(-1)

    if sol == 1:
        clf = SVC(kernel=Kernels[int(sol)], degree=8)
        BS = BS
    else:
        clf = SVC(kernel=Kernels[int(sol)])
    pred = np.zeros(test_target.shape)

    num_samples = train_data.shape[0]
    num_batches = int(np.ceil(num_samples / BS))

    # Train in epochs
    for epoch in range(Epoch):
        print(f"Epoch {epoch + 1}/{Epoch}")
        for batch in range(num_batches):
            start = batch * BS
            end = min(start + BS, num_samples)
            batch_data = train_data[start:end]
            batch_target = train_tar[start:end]
            clf.fit(batch_data, batch_target)

    # fitting x samples and y classes
    for i in range(test_target.shape[1]):
        clf.fit(train_data.tolist(), train_target[:, i].tolist())
        Y_pred = clf.predict(test_data.tolist())
        pred[:, i] = np.asarray(Y_pred)

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = ClassificationEvaluation(test_target, pred)
    return Eval, pred

