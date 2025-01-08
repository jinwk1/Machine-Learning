import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.svm import SVC

# 데이터 불러오기
data = pd.read_csv('6.1.csv', delimiter=",", keep_default_na=False, header=None, index_col=False,
                    names=['ENTROPY', 'FileType', 'RESULT'])
features = data.loc[:, 'ENTROPY':'FileType']
X = features.values
y = data['RESULT'].values

#data = pd.read_csv('6.2.csv', delimiter=",", keep_default_na=False, header=None, index_col=False,
#                   names=['ENTROPY', 'FileSize', 'FileType', 'RESULT'])
#features = data.loc[:, 'ENTROPY':'FileSize']
#X = features.values
#y = data['RESULT'].values


#data = pd.read_csv('6.3.csv', delimiter=",", keep_default_na=False, header=None, index_col=False,
#                    names=['ENTROPY', 'FileAccessDate', 'FileCreateDate', 'FileModifyDate', 'FileSize', 'FileType', 'RESULT'  ])
#features = data.loc[:, 'ENTROPY':'FileType']
#X = features.values
#y = data['RESULT'].values


X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)

param_grid = {'C': [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]}

best_score_in_trainingset = 0
best_score_in_validset = 0
best_C = 0

for C in [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]:
    svm_model = SVC(C=C, probability=True)
    svm_model.fit(X_train, y_train)
    training_score = svm_model.score(X_train, y_train)
    valid_score = svm_model.score(X_valid, y_valid)

    if valid_score > best_score_in_validset:
        best_score_in_trainingset =  training_score
        best_score_in_validset = valid_score
        best_parameters = {'C': C}
        best_C = C

print("Best score in training set: {:.6f}".format(best_score_in_trainingset))
print("Best score in valid set: {:.6f}".format(best_score_in_validset))
print("Best parameter:{}".format(best_parameters))

clf = SVC(**best_parameters, probability=True)
clf.fit(X_trainval, y_trainval)

kfold = KFold(n_splits=5, random_state=0, shuffle=True)
cross_validation_score = cross_val_score(clf, X_train, y_train, cv=kfold)
print("[KFold][Validation][cv=5] Cross validation score: {:.6f}".format(cross_validation_score.mean()))

test_score = clf.score(X_test, y_test)
print("Test set score in best parameter: {:.6f}".format(test_score))

# 혼동 행렬
Test_pref = clf.predict(X_test)
confusion = confusion_matrix(y_test, Test_pref)

TN = confusion[0][0]
FP = confusion[0][1]
FN = confusion[1][0]
TP = confusion[1][1]

# 성능 지표 계산
# Accuracy
Accuracy = (TP + TN) / (TP + TN + FP + FN)
print("\n[ACCURACY]: {:.6f}".format(Accuracy))

# Precision
Precision = (TP) / (TP + FP)
print("[PRECISION]: {:.6f}".format(Precision))

# Recall
Recall = (TP) / (TP + FN)
print("[RECALL]: {:.6f}".format(Recall))

# F1 Score
F1score = ((Precision * Recall) / (Precision + Recall) * 2)
print("[F1-SCORE]: {:.6f}".format(F1score))

# AUC
AUC = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
print("AUC: {:.6f}".format(AUC))