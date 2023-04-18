import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

class Preprocessor:
    def __init__(self, data):
        self.data = data
        self.scalng = StandardScaler()
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.corr_features = set()



    def fit(self,X,y):
        corr_matrix = X.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    colname = corr_matrix.columns[i]
                    self.corr_features.add(colname)





    def transform(self,X,y):
        X.drop(labels=self.corr_features, axis=1, inplace=True)
        X = X.dropna(axis=1, thresh=len(X) * 0.6)
        X = self.imputer.fit_transform(X)
        X = self.scalng.fit_transform(X)
        oversample = SMOTE()
        X, y = oversample.fit_resample(X, y)
        return X,y




data = pd.read_csv("C:\\Users\\hrantb\\Downloads\\hospital_deaths_train.csv")
y = data['In-hospital_death']
X = data.drop(['In-hospital_death'], axis=1)
print(type(X))
preprocessor = Preprocessor(data)
preprocessor.fit(X,y)
preprocessor.transform(X,y)
