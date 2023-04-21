import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, BaggingClassifier, \
            GradientBoostingClassifier,AdaBoostClassifier

from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd




df = pd.read_csv("/home/gago/Documents/ACA homeworks/hospital_deaths_train.csv")
X = df.drop(columns=["In-hospital_death"])
col = X.columns
y = df["In-hospital_death"]
X = SimpleImputer(missing_values=np.nan, strategy="most_frequent").fit_transform(X)

# pipeline = make_pipeline(SimpleImputer(missing_values=np.nan, strategy="mean"),
#                          StandardScaler(),
#                           PolynomialFeatures(degree=2))

pipeline = make_pipeline(StandardScaler(),
                          PolynomialFeatures(degree=2))


# imp = KNNImputer(n_neighbors=20)
# new_x = imp.fit_transform(X)
# oversample = SMOTE()
# poly = PolynomialFeatures(degree=2)
# new_x = poly.fit_transform(new_x)
# new_x, y = oversample.fit_resample(new_x, y)
# new_df2 = pd.DataFrame(new_x, columns=poly.get_feature_names_out(X.columns))
# x_train, x_test, y_train, y_test = train_test_split(new_df2.values, y, test_size=0.25, random_state=13)


pipeline.fit(X)
poly_features = pipeline.named_steps["polynomialfeatures"]
poly_features_name = poly_features.get_feature_names_out(col)
X = pipeline.transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=15)
oversample = SMOTE()
x_train, y_train = oversample.fit_resample(x_train, y_train)
#new_df = pd.DataFrame(X_train, columns=poly_features_name)

#x_train, x_test, y_train, y_test = train_test_split(new_df.values, y, test_size=0.25, random_state=13)

# estimators = [
#       ('bagging', BaggingClassifier(n_estimators=10, random_state=13)),
#       ("random forests", RandomForestClassifier(max_depth=2, random_state=2, class_weight="balanced"))
# ]
#
# clf = StackingClassifier(estimators=estimators)
# clf.fit(x_train, y_train)
# print(clf.score(x_test, y_test))

# bgf = BaggingClassifier()
# bgf.fit(x_train, y_train)
# y_pred = bgf.predict(x_test)
# print(bgf.score(x_test, y_test)) # 0.9178571428571428
# print(bgf.score(x_train, y_train)) # 0.9952358265840877
# print(f1_score(y_pred, y_test)) # 0.9144981412639406


# print(len(y_test[y_test==1]), len(y_test[y_test==0]))
# print(len(y_train[y_train==1]), len(y_train[y_train==0]))
#

# X_sm, y_sm = smote.fit_sample(y_train, y_test)
# print(X_sm.shape)
# print(y_sm.shape)

# rfc = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=23)
# rfc.fit(x_train, y_train)
# y_pred = rfc.predict(x_test)
# print("Random_forest", rfc.score(x_train, y_train)) # 0.9975292003593891
# print("Random_forest", rfc.score(x_test, y_test)) # 0.8676923076923077
# print("Random_forest", f1_score(y_test, y_pred)) # 0.43421052631578944


# cm = confusion_matrix(y_pred, y_train)
# sns.heatmap(cm, annot=True,cmap='Blues', fmt='g')
# plt.xlabel('Predicted label')
# plt.ylabel('True label')
# plt.savefig("cm.png")

# gba = GradientBoostingClassifier(n_estimators=40, max_depth=10, random_state=14)
# gba.fit(x_train, y_train)
# y_pred = gba.predict(x_test)
# print("GradineBoosting", gba.score(x_train, y_train)) # 0.9502009825815096
# print("GradineBoosting", gba.score(x_test, y_test)) # 0.91875
# print("GradineBoosting", f1_score(y_test, y_pred)) # 0.9171974522292994

# svm = SVC(C=10**4, kernel='rbf')
# svm.fit(x_train, y_train)
# y_pred = svm.predict(x_test)
# print("Svm", svm.score(x_train, y_train)) # 1.0 (C = 10**4)
# print("Svm", svm.score(x_test, y_test)) # 0.8692307692307693 (C = 10**4)
# print("Svm", f1_score(y_pred, y_test)) # 0.28571428571428575(C = 10**4)

# ada = AdaBoostClassifier(n_estimators=50 ,random_state=12)
# ada.fit(x_train, y_train)
# y_pred = ada.predict(x_test)
# print("Adaboost",ada.score(x_train, y_train)) # 0.9301437556154537
# print("AdaBoost", ada.score(x_test, y_test)) # 0.8261538461538461
# print("AdaBoost", f1_score(y_test, y_pred)) # 0.4021164021164021

# cm = confusion_matrix(y_pred, y_test)
# sns.heatmap(cm, annot=True,cmap='Blues', fmt='g')
# plt.xlabel('True label')
# plt.ylabel('Predicted label')
# plt.savefig("cm.png")

# ada = AdaBoostClassifier()
# param = {"n_estimators": [i for i in range(20, 30)]}
# grd = GridSearchCV(ada, param)
# grd.fit(x_train, y_train)
#
# print(grd.best_params_)