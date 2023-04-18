from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd


class preprocessor:

    def __init__(self, data):
        self.data = data
        self.scaling = StandardScaler()
        self.poly = PolynomialFeatures(degree=3)
        self.imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        self.pipeline = make_pipeline(self.imputer, self.scaling, self.poly)


    def fit(self, X, y=None):
        self.pipeline.fit(X)
        self.poly_feature_names = self.poly.get_feature_names_out(X.columns)
        return self.poly_feature_names



    def transform(self, X):
        return self.pipeline.transform(X)


df = pd.read_csv("/home/gago/Documents/ACA homeworks/hospital_deaths_train.csv")
X = df.drop(columns=["In-hospital_death"])
y = df["In-hospital_death"]
print((y[y==1]).shape)
print((y[y==0]).shape)
model = preprocessor(df)
name = model.fit(X, y)
new = model.transform(X)
new_df = pd.DataFrame(new, columns=name)
print(new_df)