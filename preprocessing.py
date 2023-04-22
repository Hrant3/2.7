import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report,plot_confusion_matrix,accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
  






df = pd.read_csv("hospital_deaths_train.csv")


age_ranges = [0, 18, 35, 65,90,100]
age_labels = ['0-18', '19-35', '36-65', '66-90',"90-100"]

# Create a new column 'Age Range' using cut() function
df['Age_Range'] = pd.cut(df['Age'], bins=age_ranges, labels=age_labels)





def calc_average(num1, num2):
    """
    Calculates the average of two numbers.

    Args:
        num1 (float): The first number.
        num2 (float): The second number.

    Returns:
        float: The average of the two numbers.
    """
    average = (num1 + num2) / 2
    return average




Glucose_min = 90
Glucose_max = 130
Mg_min = 1.7
Mg_max = 2.2
K_min = 3.7
K_max = 5.2
WBC_min = 4.5
WBC_max = 11
HCO3_min = 23
HCO3_max = 29
Na_min = 136
Na_max = 145
Platelets_min = 150
Platelets_max = 400
HCT_min = 36
HCT_max = 50
Creatinine_min = 0.5
Creatinine_max = 1.3
BUN_min = 7
BUN_max = 20
HR_min = 60
HR_max = 100
DiasABP_max = 80
SysABP_max = 120
MAP_min = 70
MAP_max = 100
PaO2_min = 75
PaO2_max = 100
PaCO2_min = 38
PaCO2_max = 42
pH_min = 7.35
pH_max = 7.45
NiDiasABP_max = 99
TroponinI_min = 0
TroponinI_max = 0.04
TroponinT_min = 0
TroponinT_max = 0.01
Cholesterol_max = 200
Resprate_min = 12
Resprate_max = 18
Albumin_max = 2
Alp_min = 44
Alp_max = 147
Bilirubin_min = 0.1
Bilirubin_max = 1.2
Alt_min = 7
Alt_max = 56
AST_min = 10
AST_max = 40
SaO2_min = 94
SaO2_max = 99
Lactate_max = 1


Glucose_norma = calc_average(Glucose_min,Glucose_max)
Mg_norma = calc_average(Mg_min,Mg_max)
K_norma= calc_average(K_min,K_max)
WBC_norma= calc_average(WBC_min,WBC_max)
HCO3_norma = calc_average(HCO3_min,HCO3_max)
Na_norma = calc_average(Na_min,Na_max)
Platelets_norma = calc_average(Platelets_min,Platelets_max)
HCT_norma = calc_average(HCT_min,HCT_max)
Creatinine_norma = calc_average(Creatinine_min,Creatinine_max)
BUN_norma =  calc_average(BUN_min,BUN_max)
GCS_norma = 14
HR_norma = calc_average(HR_min,HR_max)
DiasABP_norma = DiasABP_max
SysABP_norma = SysABP_max
MAP_norma = calc_average(MAP_min,MAP_max)
PaO2_norma = calc_average(PaO2_min,PaO2_max)
PaCO2_norma = calc_average(PaCO2_min,PaCO2_max)
pH_norma = calc_average(pH_min,pH_max)
NIDiasABP_norma = DiasABP_norma
NISysABP_norma = SysABP_max
NIMAP_norma = MAP_norma
TroponinI_norma = calc_average(TroponinI_min,TroponinI_max)
TroponinT_norma = calc_average(TroponinT_min,TroponinT_max)
Cholesterol_norma = Cholesterol_max
Resprate_norma = calc_average(Resprate_min,Resprate_max)
Albumin_norma = Albumin_max
ALP_norma = calc_average(Alp_min,Alp_max)
Bilirubin_norma = calc_average(Bilirubin_min,Bilirubin_max)
ALT_norma = calc_average(Alt_min,Alt_max)
AST_norma = calc_average(AST_min,AST_max)
SaO2_norma = calc_average(SaO2_min,SaO2_max)
Lactate_norma = Lactate_max








var_names = ['Glucose_norma','Mg_norma','K_norma','WBC_norma','HCO3_norma','HCO3_norma',
            'Na_norma','Platelets_norma','HCT_norma','Creatinine_norma','BUN_norma','GCS_norma',
             'HR_norma', 'DiasABP_norma', 'SysABP_norma','MAP_norma','PaO2_norma','PaCO2_norma','pH_norma','NIDiasABP_norma',
            'NISysABP_norma','NIMAP_norma','TroponinI_norma','TroponinT_norma','Cholesterol_norma','Resprate_norma',
            'Albumin_norma','ALP_norma','Bilirubin_norma','ALT_norma','AST_norma','SaO2_norma',
            'Lactate_norma']

values = [Glucose_norma,Mg_norma,K_norma,WBC_norma,HCO3_norma,HCO3_norma,
            Na_norma,Platelets_norma,HCT_norma,Creatinine_norma,BUN_norma,GCS_norma,
             HR_norma, DiasABP_norma, SysABP_norma,MAP_norma,PaO2_norma,PaCO2_norma,pH_norma,NIDiasABP_norma,
            NISysABP_norma,NIMAP_norma,TroponinI_norma,TroponinT_norma,Cholesterol_norma,Resprate_norma,
            Albumin_norma,ALP_norma,Bilirubin_norma,ALT_norma,AST_norma,SaO2_norma,
            Lactate_norma]

my_dict = {var_name: value for var_name, value in zip(var_names, values)}





nan_10s = ['Glucose_last', 'Glucose_first', 
'Glucose_highest', 'Glucose_median', 
'Glucose_lowest', 'Mg_last', 'Mg_first',
'K_last', 'K_first', 'WBC_last',
'WBC_first', 'HCO3_last', 'HCO3_first',
'Na_first', 'Na_last', 'Platelets_first',
'Platelets_last', 'HCT_first', 'HCT_last','Creatinine_first',
'Creatinine_last', 'BUN_last', 'BUN_first',
'GCS_first', 'GCS_last','GCS_median', 'GCS_highest',
'GCS_lowest', 'HR_lowest', 'HR_first', 'HR_highest', 
'HR_last', 'HR_median']




fill_values_10 = ['Glucose_norma','Glucose_norma','Glucose_norma','Glucose_norma','Glucose_norma',
             'Mg_norma','Mg_norma','K_norma','K_norma','WBC_norma','WBC_norma','HCO3_norma','HCO3_norma',
            'Na_norma','Na_norma','Platelets_norma','Platelets_norma','HCT_norma','HCT_norma','Creatinine_norma',
             'Creatinine_norma','BUN_norma','BUN_norma','GCS_norma','GCS_norma','GCS_norma','GCS_norma','GCS_norma',
             'HR_norma','HR_norma','HR_norma','HR_norma','HR_norma']





# Loop through columns and fill NaN values with specific fill values
for col, fill_val in zip(nan_10s, fill_values_10):
    df[col].fillna(my_dict[fill_val], inplace=True)







nan_40s = ['DiasABP_median', 'DiasABP_first',
'DiasABP_lowest', 'DiasABP_highest', 'DiasABP_last',
'SysABP_first', 'SysABP_last', 'MAP_median', 'MAP_highest',
'MAP_first', 'MAP_lowest', 'MAP_last', 'PaO2_first', 'PaO2_last','PaCO2_first',
'PaCO2_last', 'pH_first', 'pH_last', 'NIMAP_last', 
'NIMAP_highest', 'NIMAP_first', 'NIMAP_lowest', 'NIMAP_median',
'NIDiasABP_median', 'NIDiasABP_lowest', 'NIDiasABP_first', 
'NIDiasABP_last', 'NIDiasABP_highest', 'NISysABP_highest', 
'NISysABP_lowest', 'NISysABP_first', 'NISysABP_median', 'NISysABP_last']





fill_values_40 = ['DiasABP_norma','DiasABP_norma','DiasABP_norma','DiasABP_norma','DiasABP_norma',
             'SysABP_norma','SysABP_norma','MAP_norma','MAP_norma','MAP_norma','MAP_norma','MAP_norma',
             'PaO2_norma','PaO2_norma','PaCO2_norma','PaCO2_norma','pH_norma','pH_norma',
             'NIMAP_norma','NIMAP_norma','NIMAP_norma','NIMAP_norma','NIMAP_norma',
             'NIDiasABP_norma','NIDiasABP_norma','NIDiasABP_norma','NIDiasABP_norma','NIDiasABP_norma',
             'NISysABP_norma','NISysABP_norma','NISysABP_norma','NISysABP_norma','NISysABP_norma']


for col, fill_val in zip(nan_40s, fill_values_40):
    df[col].fillna(my_dict[fill_val], inplace=True)



nan_95s = ['TroponinI_first', 'TroponinI_last', 'Cholesterol_first', 
'Cholesterol_last', 'TroponinT_first', 'TroponinT_last',
'RespRate_last', 'RespRate_median', 'RespRate_lowest', 
'RespRate_highest', 'RespRate_first', 'Albumin_last',
'Albumin_first', 'ALP_last', 'ALP_first', 'Bilirubin_first',
'Bilirubin_last', 'ALT_first', 'ALT_last', 'AST_first', 'AST_last', 
'SaO2_last', 'SaO2_lowest', 'SaO2_first', 'SaO2_highest', 'SaO2_median', 
'Lactate_last', 'Lactate_first']


fill_values_95 = ['TroponinI_norma','TroponinI_norma','Cholesterol_norma','Cholesterol_norma',
                  'TroponinT_norma','TroponinT_norma','Resprate_norma','Resprate_norma','Resprate_norma',
                  'Resprate_norma','Resprate_norma', 'Albumin_norma', 'Albumin_norma', 'ALP_norma', 'ALP_norma',
                  'Bilirubin_norma','Bilirubin_norma','ALT_norma','ALT_norma','AST_norma','AST_norma',
                  'SaO2_norma','SaO2_norma','SaO2_norma','SaO2_norma','SaO2_norma','Lactate_norma','Lactate_norma'
                  ]




for col, fill_val in zip(nan_95s, fill_values_95):
    df[col].fillna(my_dict[fill_val], inplace=True)


#temperature

fill_with_temp_norm = ['Temp_highest',
 'Temp_lowest',
 'Temp_median',
 'Temp_first',
 'Temp_last',
 ]

Temp_norm = 36.7

for col in fill_with_temp_norm:
    df[col].fillna(Temp_norm, inplace=True)


#fill with 0, assuming that  the patient have not been passed the given procedure



fill_with_0 = ['MechVentLast8Hour',
 'FiO2_first',
 'FiO2_last', 
 'MechVentStartTime',
 'MechVentDuration']


for col in fill_with_0:
    df[col].fillna(0, inplace=True)

#columns that need to be filled with  means by age range


fill_with_means_by_age_range = ['Height','Weight','Weight_last','Weight_first','UrineOutputSum']

for col in fill_with_means_by_age_range:

    df[col] = df[col].fillna(df.groupby(['Age_Range'])[col].transform('mean'))



# fill randomly with 0 and 1

fill_randomly = ['Gender']

for col in fill_randomly:
    mask = df[col].isna()
    n_na = mask.sum()
    df[col] = df[col].fillna(pd.Series(np.random.randint(0, 2, size=n_na), index=df.index[mask]))


final_data = df

medians = final_data.filter(regex = "_median")
lasts = final_data.filter(regex = "_last")
others = final_data[['In-hospital_death','Age','Gender',"Height","Weight","CCU","CSRU","SICU",
 'MechVentStartTime','MechVentDuration','MechVentLast8Hour','UrineOutputSum']]



result = pd.concat([others,medians,lasts], axis=1)


X = result.drop(['In-hospital_death'], axis=1)
y = result['In-hospital_death']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

from imblearn.over_sampling import SMOTE

# Create an instance of SMOTE
sm = SMOTE()
scaler =  MinMaxScaler()

# Apply Min-Max scaling to train data
train_data_scaled = scaler.fit_transform(X_train)
scaled_train_data= pd.DataFrame(train_data_scaled, columns=X_train.columns)


X_train = scaled_train_data

# Apply Min-Max scaling to test data
test_data_scaled = scaler.transform(X_test)
scaled_test_data = pd.DataFrame(test_data_scaled, columns=X_test.columns)


X_test = scaled_test_data





# Apply SMOTE to the training data
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)


X_train = X_train_resampled
y_train = y_train_resampled




# from sklearn.model_selection import GridSearchCV

# n_estimators=[64,100,128,200,250,300]
# max_features= [2,3,4,5]
# bootstrap = [True,False]
# oob_score = [True,False]

# param_grid = {'n_estimators':n_estimators,
#              'max_features':max_features,
#              'bootstrap':bootstrap,
#              'oob_score':oob_score}  # Note, oob_score only makes sense when bootstrap=True!

# rfc = RandomForestClassifier()
# grid = GridSearchCV(rfc,param_grid)