import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 데이터 불러오기
data = pd.read_csv('6.1.csv', delimiter=",", keep_default_na=False, header=None, index_col=False,
                   names=['ENTROPY', 'FileType', 'RESULT'])
features = data.loc[:, 'ENTROPY':'FileType']
X = features.values
y = data['RESULT'].values

#data = pd.read_csv('6.2.csv', delimiter=",", keep_default_na=False, header=None, index_col=False,
#                   names=['ENTROPY', 'FileType', 'FileSize', 'RESULT'])
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

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

best_score_in_trainingset = 0
best_score_in_validset = 0
best_max_iter = 0
best_alpha = 0

for max_iter in [1000, 10000, 100000, 1000000]:
    for alpha in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]:
        lasso_alpha = MLPClassifier(max_iter=max_iter, random_state=0, alpha=alpha, tol=0.0001)
        lasso_alpha.fit(X_train, y_train)
        training_score = lasso_alpha.score(X_train, y_train)
        valid_score = lasso_alpha.score(X_valid, y_valid)

        if valid_score > best_score_in_validset:
                best_score_in_trainingset =  training_score
                best_score_in_validset = valid_score
                best_parameters = {'max_iter': max_iter, 'alpha': alpha}
                best_max_iter = max_iter
                best_alpha = alpha

print("Best score in training set: {:.3f}".format(best_score_in_trainingset))
print("Best score in valid set: {:.3f}".format(best_score_in_validset))
print("Best parameter:{}".format(best_parameters))

clf = MLPClassifier(**best_parameters)
clf.fit(X_trainval, y_trainval)

kfold = KFold(n_splits=5, random_state=0, shuffle=True)
cross_validation_score = cross_val_score(clf, X_train, y_train, cv=kfold)
print("[KFold][Validation][cv=5] Cross validation score: {:.5f}".format(cross_validation_score.mean()))

test_score = clf.score(X_test, y_test)
print("Test set score in best parameter: {:.3f}".format(test_score))


from sklearn.metrics import confusion_matrix


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
print("\n[ACCURACY]: {:.3f}".format(Accuracy))

# Precision
Precision = (TP) / (TP + FP)
print("[PRECISION]: {:.3f}".format(Precision))

# Recall
Recall = (TP) / (TP + FN)
print("[RECALL]: {:.3f}".format(Recall))

# F1 Score
F1score = ((Precision * Recall) / (Precision + Recall) * 2)
print("[F1-SCORE]: {:.3f}".format(F1score))

# AUC
AUC = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
print("AUC: {:.5f}".format(AUC))