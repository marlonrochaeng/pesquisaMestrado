from libsvm.svmutil import svm_train, svm_predict

class SVM_info():

    def __init__(self, etc, ind) -> None:
        self.etc = etc
        self.ind = ind

    def create_classifier(self):
        from sklearn import svm
        self.clf = svm.SVC(kernel='rbf', probability=True)

    def train(self):
        self.svc = self.clf.fit(self.etc, self.ind)
        return self.svc

class LibSVM_info():

    def __init__(self, etc, ind) -> None:
        self.etc = etc
        self.ind = ind

    def train(self):
        self.m = svm_train(self.ind, self.etc, '-t 2 -b 1 -s 0')
        return self.m
    
    def get_label(self,labels, model, info):
        labels = svm_predict(labels, info, model)
        return labels
        