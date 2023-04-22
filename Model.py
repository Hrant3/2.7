from sklearn.ensemble import AdaBoostClassifier
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.metrics import f1_score
class Model:
    def __init__(self,model):
        self.model = model
        self.threshold = 0.5
        self.classifiers = {
            'AdaBoost': AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42),
            'SVM': SVC(gamma='auto', probability=True, random_state=42),
            'NaiveBayes': MultinomialNB(),
            'LogisticRegression': LogisticRegression(solver='lbfgs', random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=3),
            'DecisionTree': DecisionTreeClassifier(max_depth=5),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)}


    def fit(self, X, y):
        clf = self.classifiers[self.model]
        clf.fit(X, y)
        self.model = clf



        # self.model = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        # self.model.fit(X.values, y)
        # joblib.dump(self.model, 'model.pkl')

       # if self.model == AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42):

           # Adaboost_param_grid = {"n_estimators": [100, 200],"learning_rate": [0.1, 0.5]}
           # self.model = GridSearchCV(self.model(), Adaboost_param_grid, cv=5, scoring='f1')
           # param = self.model.best_params_
         #   self.model = AdaBoostClassifier(**param)
            


    def predict(self, X):
        return self.model.predict(X)
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    def score(self, X, y):
        return self.model.score(X, y)

    # def get_threshold(self, X, y):
    #     # get the threshold that gives the best F1 score
    #     thresholds = np.linspace(0, 1, 25)
    #     f1_scores = []
    #     for threshold in thresholds:
    #         y_pred = self.predict_proba(X)[:, 1] > threshold
    #         f1_scores.append(f1_score(y, y_pred))
    #     return thresholds[np.argmax(f1_scores)]