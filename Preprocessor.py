import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
#import polynomial features
from sklearn.preprocessing import PolynomialFeatures


class Preprocessor:
    def __init__(self):
        self.scalng = StandardScaler()
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.corr_features = set()
        self.X = None
        self.y = None
        self.mean_values = None
        self.mean = None
        self.std = None
        self.cols_to_drop = None
        self.poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    def fit(self,data):
       self.corr_comp(data)
    def transform(self,X,test=False):
        if test:
            self.X = X
          #  self.X = data
            self.X.drop(labels=self.corr_features, axis=1, inplace=True)
            self.X = self.X.drop(self.cols_to_drop, axis=1)
            self.X = self.X.fillna(self.mean_values)
            self.X = (self.X - self.mean) / self.std
            return self.X


        else:
            X.drop(labels=self.corr_features, axis=1, inplace=True)
            self.cols_to_drop = X.columns[X.isna().mean() > 0.8]
            X = X.drop(self.cols_to_drop, axis=1)
          #  X = X.dropna(axis=1, thresh=len(X) * 0.6)
            self.mean_values = X.mean()
            X = X.fillna(self.mean_values)
            X = self.scalng.fit_transform(X)
            self.mean = self.scalng.mean_
            self.std = self.scalng.scale_
            return X
    def corr_comp(self,X):
        corr_matrix = X.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    colname = corr_matrix.columns[i]
                    self.corr_features.add(colname)





# data = pd.read_csv("C:\\Users\\hrantb\\Downloads\\hospital_deaths_train.csv")
# y = data['In-hospital_death']
# X = data.drop(['In-hospital_death'], axis=1)
# print(type(X))
# preprocessor = Preprocessor(data)
# preprocessor.fit(data)
# preprocessor.transform(data)
