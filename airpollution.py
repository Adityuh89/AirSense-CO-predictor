import pandas  as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV

path = r'c:\Users\Aditya\Downloads\air+quality\AirQualityUCI.csv'
df = pd.read_csv(path, sep=';', decimal=',', engine='python')
#print(df.head())
#print(df.columns)

#print(df.isnull().sum()/len(df)*100)
df = df.dropna(how='all', axis=1).dropna(how='all', axis=0)
df.replace(-200, np.nan, inplace=True)
df= df.dropna(subset=['CO(GT)'])
#plt.figure(figsize=(20,10))
#sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
#plt.title("missing values heatmap")
#plt.show()
#------------------------------------------------
y=df['CO(GT)'] 
#predicting CO concentration in the air

#where PT08.S1(CO)- hourly averaged senor response nominally CO targeted
#RH RELATIVE HUMIDTY T TEMPERATURE AH ABSOLUTE HUMIDTY

df['DateTime'] = pd.to_datetime(
    df['Date'] + ' ' + df['Time'], 
    format='%d/%m/%Y %H.%M.%S'
)
df['Hour']=df['DateTime'].dt.hour
df['Month']=df['DateTime'].dt.month
df['dayofweek']=df['DateTime'].dt.dayofweek
Features=['PT08.S1(CO)','RH','T','AH','dayofweek','Month','Hour']
X=df[Features]


numerical_columns=['RH', 'T', 'AH','PT08.S1(CO)','dayofweek','Month','Hour']

print(numerical_columns)

X_train, X_val, y_train, y_val=train_test_split(
    X,
    y,
    test_size= 0.2,
    random_state=42)
#print('Training DATA')
#print(X_train.shape)
#print('Validation DATA')
#print(X_val.shape)

numerical_transformer=Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])
preprocessor= ColumnTransformer([
    ('num', numerical_transformer, numerical_columns),
    
])
model={
    'regressor': RandomForestRegressor(random_state=42),
    'xgboost': XGBRegressor( random_state=42),
    'linear_reg': LinearRegression()
}
para_grids={
    'regressor':{
        'models__n_estimators':[100, 200, 300],
        'models__max_depth':[5, 10, 15]},
        'xgboost':{
            'models__n_estimators':[100,200,300],
            'models__max_depth':[3, 5, 8],
            'models__learning_rate':[0.01,0.03,0.05],
            'models__min_child_weight': [1, 3, 5, 7]
        },
        'linear_reg':{}
    }

kf= KFold(n_splits=5, shuffle=True, random_state=42)
for name, model_obj in model.items():
    pipeline= Pipeline([
        ('preprocessor', preprocessor),
        ('models', model_obj)
    ])
    grid_search=GridSearchCV(
        estimator= pipeline, 
        param_grid=para_grids[name], 
        cv=kf, 
        scoring='r2',
        n_jobs=-1)
    grid_search.fit(X_train,y_train)
    print(f"models name: {name}")
    print(f"Winner Params: {grid_search.best_params_}")
    print(f"Optimized R2 Score: {grid_search.best_score_:.2%}")
    #print(f"all scores: {grid_search.cv_results_}")
    
    finalbestmodel= grid_search.best_estimator_
    predictions= finalbestmodel.predict(X_val)
    print(f"Validation R2:{ r2_score(y_val, predictions):.2%}")

import joblib
joblib.dump(finalbestmodel,'airquality.pkl')
print("Model saved")











 


