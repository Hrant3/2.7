from Model import Model
from Preprocessor import Preprocessor
import joblib
import json
import pandas as pd

import argparse
from sklearn.model_selection import train_test_split
class Pipeline:
    def __init__(self,model):
        self.model = Model(model)
        self.preprocessor = Preprocessor()


    def run(self, X,y, test=False):
        if test:
            # load preprocessor and model
            self.preprocessor = joblib.load('preprocessor.pkl')
            self.model = joblib.load('model.pkl')
            X = self.preprocessor.transform(X,test=True)
          #  print(X.columns)
           # predict_proba = self.model.predict_proba(X)
            score = self.model.score(X,y)
            print(score)
            threshold = self.model.threshold
            # jsonFile = {'predict_proba': self.model.predict_proba, 'threshold': threshold}
            #
            # with open('predictions.json', 'w') as f:
            #     json.dump(jsonFile, f)


           # threshold = self.model.get_threshold(X,y)
            with open('predictions.json', 'w') as f:
                json.dump({'predict_probas': self.model.predict_proba(X).tolist(), 'threshold': threshold}, f)
        else:
            self.preprocessor.fit(X)
            X= self.preprocessor.transform(X)
            self.model.fit(X,y)
            joblib.dump(self.preprocessor, 'preprocessor.pkl')
            joblib.dump(self.model, 'model.pkl')


        # call preprocessor and model for training
        # save preprocessor and model for future testing
# data = pd.read_csv("C:\\Users\\hrantb\\Downloads\\hospital_deaths_train.csv")
# print(data)
# X = data.drop(['In-hospital_death'], axis=1)
# y = data['In-hospital_death']
#
#
#
# #doing train test split to test the model
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# joblib.dump(X_test, 'X_test.pkl')
# joblib.dump(y_test, 'y_test.pkl')
# #Pipeline(X).run(X_test,y_test,test=True)
# #Pipeline(X).run(X_train,y_train,test=False)
# Pipeline(X).run(X_test,y_test,test=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',help="Path to data file", required=True)
    parser.add_argument('--model',help="Model to use", required=False,default='None')
    parser.add_argument('--inference',type=bool,help="Test the model", required=False,default=False)
    args = parser.parse_args()
    data_path = args.data_path
    inference = args.inference
    model = args.model
    data = pd.read_csv(data_path)
    X = data.drop(['In-hospital_death'], axis=1)
    y = data['In-hospital_death']
    pipeline = Pipeline(model)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #pipeline.run(X_train, y_train, test=inference)
    pipeline.run(X_test, y_test, test=inference)
    # print(type(model))
    # if inference:
    #     Pipeline(model).run(X_train,y_train,test=True)
    # else:
    #     Pipeline(model).run(X_test,y_test,test=False)

if __name__ == '__main__':
    main()