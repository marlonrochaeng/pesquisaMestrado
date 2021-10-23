from libsvm.svmutil import svm_train, svm_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class SVM_info():

    def __init__(self, etc, ind) -> None:
        self.etc = etc
        self.ind = ind

    def create_classifier(self):
        from sklearn import svm
        from sklearn.calibration import CalibratedClassifierCV
        self.clf = svm.SVC(kernel='rbf', probability=True, random_state=0)
        self.clf = CalibratedClassifierCV(self.clf, method='sigmoid', cv=5, n_jobs=4)

    def train(self):
        self.svc = self.clf.fit(self.etc, self.ind)
        #print(self.svc.)
        #input()
        return self.svc

class LibSVM_info():

    def __init__(self, etc, ind) -> None:
        self.etc = etc
        self.ind = ind

    def train(self):
        self.m = svm_train(self.ind, self.etc, '-t 2 -b 1 -s 0')
        print(self.m.probB)
        return self.m
    
    def get_label(self,labels, model, info):
        labels = svm_predict(labels, info, model)
        return labels

class RandomForestClass():
    def __init__(self, etc, ind) -> None:
        self.etc = etc
        self.ind = ind
    
    def create(self):
        self.rfc = RandomForestClassifier()
    
    def fit(self):
        self.rfc.fit(self.etc, self.ind)
        param_grid = [
            {'n_estimators': [10, 25, 100], 'max_features': [5, 10], 
            'max_depth': [10, 50, None], 'bootstrap': [True, False]}
            ]
        self.grid_search = GridSearchCV(self.rfc, param_grid, cv=10, scoring='neg_mean_squared_error')
        self.grid_search.fit(self.etc, self.ind)

    def pred(self, i):
        return self.rfc.predict(i)

    def proba(self, i):
        return self.rfc.predict_proba(i)