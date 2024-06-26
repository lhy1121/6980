#import packages
import pandas as pd
import numpy as np
from xgboost import XGBRegressor as XGBR
import matplotlib.pyplot as plt
import xgboost
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

folder_path = "./system/engines/model/Xgboost"  
#data cleaning functions
def processing_data(train_data):
    for column in list(train_data.columns[train_data.isnull().sum()>0]):
        mean=train_data[column].mean()
        median=train_data[column].median()
        sigma=train_data[column].std()
        train_data[column].fillna(median,inplace=True)
        feature_celling=mean+3*sigma
        feature_floor=mean-3*sigma
        train_data[column]=np.clip(train_data[column],feature_floor,feature_celling)
    return train_data

def zscore_data(train_data):
    columns=train_data.columns
    for feature in columns[2:]:
        feature_mean=train_data[feature].mean()
        feature_std=train_data[feature].std()
        train_data[feature]=(train_data[feature]-feature_mean)/feature_std
    return train_data

def xgb_model_training(data,target,start_year,train_years,validation_years,test_years,subsample,learning_rate,max_depth,X_test):

    #let the start date be the datetime
    train_start=pd.to_datetime(str(start_year))
    #use a period of time as training dates
    train_end=train_start+pd.DateOffset(years=train_years)-pd.DateOffset(days=1)
    #use the validation dates to show the scoring of prediction of model
    validation_start=train_end+pd.DateOffset(days=1)
    validation_end=validation_start+pd.DateOffset(years=validation_years)-pd.DateOffset(days=1)
    '''
    test_start=train_end+pd.DateOffset(days=1)
    test_end=test_start+pd.DateOffset(years=test_years)-pd.DateOffset(days=1)
    '''
    
    feature=data.columns.tolist()
    X_train=data.loc[train_start:train_end][feature].values
    y_train=target.loc[train_start:train_end].values
    X_validation=data.loc[validation_start:validation_end][feature].values
    y_validation=target.loc[validation_start:validation_end].values
    
    print("start model training:")
    xgb_reg=XGBR(subsample=subsample,learning_rate=learning_rate,max_depth=max_depth)
    xgb_reg.fit(X_train,y_train)
    y_pred=xgb_reg.predict(X_validation)
    
    #calculate mse of validation set
    va_y=y_validation.reshape(-1,1)
    va_y_pred = y_pred.reshape(-1,1)
    mean_square_error=np.mean((va_y-va_y_pred)**2)
    
    y_pred_test=xgb_reg.predict(X_test)
    va_y_pred = np.vstack((va_y_pred, y_pred_test))
    return mean_square_error,va_y,va_y_pred,xgb_reg

def xgb_model_testing(data,target,start_year,train_years,validation_years,test_years,subsample,X_test,city,feature_name):
    train_start=pd.to_datetime(str(start_year))
    train_end=train_start+pd.DateOffset(years=train_years)-pd.DateOffset(days=1)
    validation_start=train_end+pd.DateOffset(days=1)
    validation_end=validation_start+pd.DateOffset(years=validation_years)-pd.DateOffset(days=1)
    test_start=train_end+pd.DateOffset(days=1)
    test_end=test_start+pd.DateOffset(years=test_years)-pd.DateOffset(days=1)   
    feature=data.columns.tolist()
    X_validation=data.loc[validation_start:validation_end][feature].values
    y_validation=target.loc[validation_start:validation_end].values  
    print("start model training:")
    xgb_reg=XGBR()
    model_file = "Xgboost"+'-'+city+'-'+feature_name+'.bin'
    model_path = os.path.join(folder_path, model_file)
    xgb_reg.load_model(model_path)
    y_pred=xgb_reg.predict(X_validation)
    y_test = y_validation
    #calculate prediction
    va_y=y_validation.reshape(-1,1)
    va_y_pred=y_pred.reshape(-1,1)

    #test scoring illustrated there:MSE R2 and mean absolute mean absolute error.
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    
    y_pred_test=xgb_reg.predict(X_test)
    va_y_pred = np.vstack((va_y_pred, y_pred_test))
    return mse,va_y,va_y_pred,r2,mae
    #Set year as index

def xgboost_func(source_data,city, feature_name,train = 1):
    source_data['year'] = pd.to_datetime(source_data['year'], format='%Y')
    source_data = source_data.sort_values(by=['year'])
    source_data = source_data.set_index(['year',])
    train_data=source_data[source_data['country']==city]

    target_1=pd.DataFrame(train_data[feature_name].values,index=train_data[feature_name].index - pd.DateOffset(years=1),columns=[feature_name])
    target_1=target_1.drop(target_1.index[0])

    processed_train_data=processing_data(train_data)
    processed_train_data=zscore_data(processed_train_data)
    processed_train_data=processed_train_data.iloc[:,2:]    
    X_test = processed_train_data.iloc[-1].values
    X_test = X_test.reshape(1,-1)
    processed_train_data = processed_train_data.drop(processed_train_data.index[-1])
    
    if train == 1:
        subsample_list=[0.8,0.85,0.9]
        learning_rate_list=[0.01,0.05,0.1,0.2]
        max_depth_list=[2,3,5,7,10]
        MSE=1000000
        combination=[0,0,0]
        for x in range(0,2):
            for y in range(0,3):
                for z in range(0,3):
                    subsample=subsample_list[x]
                    learning_rate=learning_rate_list[y]
                    max_depth=max_depth_list[z]
                    # print("subsample:",subsample)
                    # print("learning_rate:",learning_rate)
                    # print("max_depth:",max_depth)
                    start_year=1932
                    train_years=int(len(processed_train_data)*subsample)
                    validation_years=len(processed_train_data)-train_years
                    test_years = 0                    
                    result=xgb_model_training(processed_train_data,target_1,start_year,train_years,validation_years,test_years,subsample,learning_rate,max_depth,X_test) 
                    #find best prediction:
                    if result[0]<MSE:
                        MSE = result[0]
                        va_y = result[1]
                        va_pred = result[2]
                        used_train = train_years
                        combination[0] = x
                        combination[1] = y
                        combination[2] = z
                        fmodel = result[3]
                        print(combination)

        model_file = "Xgboost"+'-'+city+'-'+feature_name+'.bin'
        model_path = os.path.join(folder_path, model_file)
        fmodel.save_model(model_path)
    else:
        subsample=0.7
        print("subsample:",subsample)
        start_year=1932
        train_years=int(len(processed_train_data)*subsample)
        validation_years=len(processed_train_data)-train_years
        test_years = 0
        result = xgb_model_testing(processed_train_data,target_1,start_year,train_years,validation_years,test_years,subsample,X_test,city=city,feature_name=feature_name)
        MSE = result[0]
        va_y = result[1]
        va_pred = result[2]
        R2 = result[3]
        MAE = result[4]
        used_train = train_years

    #vasualization:
    x = range(0,len(va_pred))
    years = []
    start_year = 1933
    for i in x:
        years.append(pd.to_datetime(str(start_year+used_train))+pd.DateOffset(years=i))
    #fig,ax = plt.subplots(figsize = (10,6))
    plt.plot(years,va_pred,label = 'model test')
    x = range(1,len(va_pred)+1)
    plt.plot(years[:-1],va_y,label = 'real value')
    plt.xlabel('last predicted years')
    plt.ylabel('price')
    plt.legend()
    plt.title("Xgboost Model")
    plt.show()
    print('train model totally using ',used_train,'pieces of data','mse:',MSE)
    return plt,va_pred,years,MSE,R2,MAE