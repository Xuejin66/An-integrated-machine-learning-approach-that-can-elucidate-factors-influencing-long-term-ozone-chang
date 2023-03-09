import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from multiprocessing.pool import ApplyResult
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import random
import multiprocessing 
from scipy import stats
import math
# from chinese_calendar import is_workday, is_holiday
import shap
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import accuracy_score
import pickle
import os

'''Create function'''

def cal_RH(dew,tmp,method='method1'):

    if method == 'method1':
    # source:https://earthscience.stackexchange.com/questions/16570/how-to-calculate-relative-humidity-from-temperature-dew-point-and-pressure
        rh = 100*math.pow(10,(7.591386*(dew/(dew + 240.7263) - tmp/(tmp + 240.7263))))
    elif method == 'method2':
    # source:https://bmcnoldy.rsmas.miami.edu/Humidity.html
        rh = 100*(math.exp((17.625*dew)/(243.04 + dew))/math.exp((17.625*tmp)/(243.04 + tmp)))
    else:
        print('Wrong method!')
    return rh

# Data pre-processing
def prep_data(aqData,metData,timevar):
    combinedData = pd.merge(aqData,metData,on=timevar,how='left')
    # aqData
    aqData['date'] = aqData[timevar].dt.date
    # aqData['unix_time'] = aqData['date'].apply(lambda x:(pd.to_datetime(x) - pd.Timestamp("1970-01-01")) \
    #                                         // pd.Timedelta('1s'))
    aqData['unix_time'] = aqData[timevar].apply(lambda x:(x - pd.Timestamp("1970-01-01 00:00:00")) \
                                            // pd.Timedelta('1s'))
    aqData['dayofweek'] = aqData[timevar].dt.dayofweek
    aqData['dayofyear'] = aqData[timevar].dt.dayofyear
    aqData['hour'] = aqData[timevar].dt.hour
    aqData['month'] = aqData[timevar].dt.month
    # metData
    metData['date'] = metData[timevar].dt.date
    # metData['unix_time'] = metData['date'].apply(lambda x:(pd.to_datetime(x) - pd.Timestamp("1970-01-01")) \
    #                                         // pd.Timedelta('1s'))
    metData['unix_time'] = metData[timevar].apply(lambda x:(x - pd.Timestamp("1970-01-01 00:00:00")) \
                                            // pd.Timedelta('1s'))
    metData['dayofweek'] = metData[timevar].dt.dayofweek
    metData['dayofyear'] = metData[timevar].dt.dayofyear
    metData['hour'] = metData[timevar].dt.hour
    metData['month'] = metData[timevar].dt.month
    # combinedData
    combinedData['date'] = combinedData[timevar].dt.date
    # combinedData['unix_time'] = combinedData['date'].apply(lambda x:(pd.to_datetime(x) - pd.Timestamp("1970-01-01")) \
    #                                         // pd.Timedelta('1s'))
    combinedData['unix_time'] = combinedData[timevar].apply(lambda x:(x - pd.Timestamp("1970-01-01 00:00:00")) \
                                            // pd.Timedelta('1s'))
    combinedData['dayofweek'] = combinedData[timevar].dt.dayofweek
    combinedData['dayofyear'] = combinedData[timevar].dt.dayofyear
    combinedData['hour'] = combinedData[timevar].dt.hour
    combinedData['month'] = combinedData[timevar].dt.month

    return aqData,metData,combinedData

def train_model(combinedData,pol,met,random_seed,test_size=0.2,n_estimators=300,maxFeatures=5,minSamplesLeaf=1):
    y = combinedData[[pol]]
    x = combinedData[['unix_time','dayofyear','dayofweek','hour']+met]
    train_x,test_x,train_y,test_y=train_test_split(x,y,test_size = test_size,random_state=random_seed)
    random_forest_model=RandomForestRegressor(n_estimators=n_estimators,n_jobs=-1,max_features=maxFeatures,min_samples_leaf=minSamplesLeaf,random_state=random_seed)
    random_forest_model = random_forest_model.fit(train_x.values,train_y.values.ravel())
    random_forest_predict=random_forest_model.predict(test_x.values)

    testDataset = pd.concat([test_x,test_y,pd.Series(random_forest_predict,index=test_y.index)],axis=1)
    testDataset = testDataset.rename(columns={0:'predicts'})
    random_forest_pearson_r=stats.pearsonr(test_y.values.ravel(),random_forest_predict)
    random_forest_R2=metrics.r2_score(test_y.values.ravel(),random_forest_predict)
    random_forest_RMSE=metrics.mean_squared_error(test_y.values.ravel(),random_forest_predict)**0.5
    modelPerformance = {'pol':pol,'pearsonR':random_forest_pearson_r[0],'random_forest_R2':random_forest_R2,'random_forest_RMSE':random_forest_RMSE,'rand_seed':random_seed}
    modelPerformance = pd.DataFrame(modelPerformance,index=[0])

    importances = random_forest_model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in random_forest_model.estimators_], axis=0)
    forest_importances = pd.DataFrame([importances,std],columns=x.columns,index=['importance','std']).T

    return random_forest_model,modelPerformance,forest_importances,train_x,test_x,train_y,test_y

# Training Model
def train_model_met(combinedData,pol,met,random_seed,test_size=0.2,n_estimators=300,maxFeatures=5,minSamplesLeaf=1):
    y = combinedData[[pol]]
    x = combinedData[met]
    train_x,test_x,train_y,test_y=train_test_split(x,y,test_size = test_size,random_state=random_seed)
    random_forest_model=RandomForestRegressor(n_estimators=n_estimators,n_jobs=-1,max_features=maxFeatures,min_samples_leaf=minSamplesLeaf,random_state=random_seed)
    random_forest_model = random_forest_model.fit(train_x.values,train_y.values.ravel())
    random_forest_predict=random_forest_model.predict(test_x.values)

    testDataset = pd.concat([test_x,test_y,pd.Series(random_forest_predict,index=test_y.index)],axis=1)
    testDataset = testDataset.rename(columns={0:'predicts'})
    random_forest_pearson_r=stats.pearsonr(test_y.values.ravel(),random_forest_predict)
    random_forest_R2=metrics.r2_score(test_y.values.ravel(),random_forest_predict)
    random_forest_RMSE=metrics.mean_squared_error(test_y.values.ravel(),random_forest_predict)**0.5
    modelPerformance = {'pol':pol,'pearsonR':random_forest_pearson_r[0],'random_forest_R2':random_forest_R2,'random_forest_RMSE':random_forest_RMSE,'random_seed':random_seed}
    modelPerformance = pd.DataFrame(modelPerformance,index=[0])

    importances = random_forest_model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in random_forest_model.estimators_], axis=0)
    forest_importances = pd.DataFrame([importances,std],columns=x.columns,index=['importance','std']).T

    return random_forest_model,modelPerformance,forest_importances,train_x,test_x,train_y,test_y

# Weather normalization
def normalize_weather(pol,met,timevar,aqData,metData,model):
    variables = ['unix_time','dayofyear','dayofweek','hour'] + met
    seed3 = random.randint(0,100000)
    data_sample = metData.sample(len(aqData),random_state=seed3)
    data_sample['unix_time'] = aqData['unix_time'].values
    data_sample['datetime'] = aqData[timevar].values
    data_sample['hour'] = aqData['hour'].values
    data_sample['dayofweek'] = aqData['month'].values
    data_sample['value'] = aqData[pol].values
    data_sample = data_sample.sort_values(['datetime'])
    data_in = data_sample[variables]
    predict_result = model.predict(data_in.values)

    return predict_result,data_sample

# Multi-threaded weather normalization
def normalize_weather_mtp(pol,met,timevar,aqData,metData,model,times,number_of_processors):
    pool = multiprocessing.Pool(processes=number_of_processors)
    ApplyResult = []
    for num in range(times):
        print('NO.%s SUM:%s'%(num+1,times))
        ApplyResult.append(pool.apply_async(normalize_weather, (pol,met,timevar,aqData,metData,model)))
    pool.close()
    pool.join()
    data_predicts = np.array([res.get()[0] for res in ApplyResult])
    data_sample = ApplyResult[0].get()[1]
    data = pd.DataFrame(data_predicts.T)
    data['datetime'] = data_sample['datetime'].values
    data['observation'] = data_sample['value'].values
    return data,data_sample

# Model explanation
def model_shap_values(x,model):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)
    return shap_values

def model_shap_interaction_values(x,model):
    explainer = shap.TreeExplainer(model)
    shap_interaction_values = explainer.shap_interaction_values(x)
    return shap_interaction_values

def model_shap_values_mtp(x,n,model,number_of_processors):
    pool = multiprocessing.Pool(processes=number_of_processors)
    nrows = len(x)
    times = nrows//n
    times_ = nrows%n
    if times_==0:
        times = times
    elif times_!=0:
        times = times+1
    ApplyResult = []
    for irow in np.arange(0,times):
        x_explain = x.iloc[irow*n:(irow+1)*n,:]
        ApplyResult.append(pool.apply_async(model_shap_values, (x_explain,model)))
    pool.close()
    pool.join()
    shap_values = np.array([res.get() for res in ApplyResult])
    return shap_values

def model_shap_interaction_values_mtp(x,n,model,number_of_processors):
    pool = multiprocessing.Pool(processes=number_of_processors)
    nrows = len(x)
    times = nrows//n
    times_ = nrows%n
    if times_==0:
        times = times
    elif times_!=0:
        times = times+1
    ApplyResult = []
    for irow in np.arange(0,times):
        x_explain = x.iloc[irow*n:(irow+1)*n,:]
        ApplyResult.append(pool.apply_async(model_shap_interaction_values, (x_explain,model)))
    pool.close()
    pool.join()
    shap_interaction_values = np.array([res.get() for res in ApplyResult])
    return shap_interaction_values

if __name__ == '__main__':
    sitename = 'PT'
    pol = 'O31'
    timevar = 'datetime'
    number_of_processors = 16
    times= 1000
    resultDir = 'E:\Desktop\Ozone in Shanghai\An-integrated-machine-learning-approach-that-can-elucidate-factors-influencing-long-term-ozone-chang\outputData'
    if not os.path.exists(resultDir):
        os.makedirs(resultDir)
        
    '''Air quality data pre-processing'''
    aqData_path = r'E:\Desktop\Ozone in Shanghai\An-integrated-machine-learning-approach-that-can-elucidate-factors-influencing-long-term-ozone-chang\inputData\airQuality&metData.csv'
    aqData = pd.read_csv(aqData_path,parse_dates=['datetime'])
    aqData['NOx'] = aqData['NO'] + aqData['NO2']
    aqData = aqData.loc[aqData['sitename']==sitename]
    aqData = aqData.loc[(aqData['datetime'].dt.month>3)&(aqData['datetime'].dt.month<10)]
    aqData = aqData.loc[(aqData['datetime'].dt.hour>9)&(aqData['datetime'].dt.hour<18)]
    polData = aqData.loc[:,['datetime',pol]]

    '''Meteorological data pre-processing'''
    era_path = r'E:\Desktop\Ozone in Shanghai\An-integrated-machine-learning-approach-that-can-elucidate-factors-influencing-long-term-ozone-chang\inputData\SH_ERA5_1997_2021.csv'
    metData = pd.read_csv(era_path,parse_dates=['time'])
    metData = metData.loc[(metData['time'].dt.year>2012)&(metData['time'].dt.year<2022)]
    metData = metData.loc[(metData['time'].dt.month>3)&(metData['time'].dt.month<10)]
    metData = metData.loc[(metData['time'].dt.hour>9)&(metData['time'].dt.hour<18)]
    metData = metData.rename(columns={'time':'datetime'})
    metData = metData.loc[metData['sitename']==sitename]
    metData['rh'] = metData.apply(lambda x:cal_RH(x.d2m,x.t2m),axis=1)
    metData['t2m'] = metData['t2m'] - 273.15
    metData['d2m'] = metData['d2m'] - 273.15
    metData['rh'] = metData.apply(lambda row:cal_RH(row.d2m,row.t2m,method='method2'),axis=1)
    metData['wd'] = metData.apply(lambda x:(270-(180/np.pi)*math.atan2(x.v10,x.u10))%360,axis=1)
    metData['ws'] = metData.apply(lambda x:np.sqrt((x.u10)**2+(x.v10)**2),axis=1)
    metData = metData.rename(columns={'rh':'RH','t2m':'T','ws':'WS','wd':'WD','sp':'SP','blh':'BLH','tp':'TP','ssr':'SSR','tcc':'TCC'})
    cluster = pd.read_csv(r'/data/home/xuejin/code/Traj/hysplitResults/2013_2021_traj_cluster.csv',parse_dates=['datetime'])
    cluster['cluster'] = cluster['cluster'].apply(lambda x:x.split('C')[1])
    # cluster = cluster.loc[(cluster['datetime'].dt.year>2012)&(cluster['datetime'].dt.year<2022)]
    metData = metData.merge(cluster,on=['datetime','sitename'],how='inner')
    # metData = metData.iloc[:100,:]

    '''Data processing and merging'''
    aqData = aqData.loc[aqData['datetime'].isin(metData['datetime'])]
    aqData1,metData1,combinedData = prep_data(aqData,metData,timevar='datetime')
    
    '''Model Training'''
    met = ['T','WD','WS','SSR','SP','BLH','TP','RH','TCC',
       'length','cluster',
            ]
    random_forest_seed=random.randint(0,100)
    RFmodel,modelPerformance,forest_importances,train_x,test_x,train_y,test_y = train_model(combinedData,pol,met,random_forest_seed,test_size=0.1,n_estimators=1500,maxFeatures=14,minSamplesLeaf=1)
    modelPerformance.to_csv(r'%s/modelPerformance_%s_%s.csv'%(resultDir,pol,sitename))
    forest_importances.to_csv(r'%s/feature_importances_%s_%s.csv'%(resultDir,pol,sitename))
    with open(r'%s/RFmodel_%s_%s.pkl'%(resultDir,pol,sitename), 'w+b') as fp:      # 保存模型
        pickle.dump(RFmodel, fp)  
        
    '''Weather normalization'''
    variables = ['unix_time','dayofyear','dayofweek','hour'] + met
    results = []
    for num in range(times):
        seed3 = random.randint(0,100000)
        data_sample = metData1.sample(len(aqData1),random_state=seed3)
        data_sample['unix_time'] = aqData1['unix_time'].values
        data_sample['datetime'] = aqData1[timevar].values
        data_sample['value'] = aqData1[pol].values
        data_in = data_sample[variables]
        predict_result = RFmodel.predict(data_in.values)
        results.append(predict_result)
    data_rmw = pd.DataFrame(np.array(results).T,index=aqData['datetime'])
    rmwResult = pd.DataFrame(np.array(results).mean(axis=0),index=combinedData['datetime'],columns=['rmw'])
    rmwResult['obs'] = combinedData['O31'].values
    rmwResult['wc'] = rmwResult['obs'] - rmwResult['rmw']
    rmwResult = rmwResult.reset_index()
    rmwResult.to_csv(r'%s/weather-normalized_result_%s_%s.csv'%(resultDir,pol,sitename))

    
    combinedData = combinedData.merge(rmwResult,left_on=['datetime','O31'],right_on=['datetime','obs'])
    polData = combinedData.loc[:,['datetime','wc']]
    random_forest_wc_seed=random.randint(0,100)
    RFmodel_wc,modelPerformance_wc,forest_importances_wc,trainWc_x,testWc_x,trainWc_y,testWc_y = train_model_met(combinedData,'wc',met,random_forest_wc_seed,test_size=0.1,n_estimators=1500,maxFeatures=8,minSamplesLeaf=1)
    modelPerformance_wc.to_csv(r'%s/modelPerformance_wc_%s_%s.csv'%(resultDir,pol,sitename))
    forest_importances_wc.to_csv(r'%s/feature_importances_wc_%s_%s.csv'%(resultDir,pol,sitename))
    with open(r'%s/RFmodel_wc_%s_%s.pkl'%(resultDir,pol,sitename), 'w+b') as fp:      # 保存模型
        pickle.dump(RFmodel_wc, fp)  

    '''Extracting data to be interpreted'''
    x_explain = combinedData[trainWc_x.columns]
    x_explain.to_csv(r'%s/x_explain_%s_%s.csv'%(resultDir,pol,sitename),index=False)

    '''Preparing to interpret model data'''
    with open(r'%s/RFmodel_wc_%s_%s.pkl'%(resultDir,pol,sitename), 'rb') as fp:
        RFmodel_wc = pickle.load(fp)
    x_explain = pd.read_csv(r'%s/x_explain_%s_%s.csv'%(resultDir,pol,sitename))

    '''Model explanation'''   
    explainer = shap.TreeExplainer(RFmodel_wc)
    with open(r'%s/RFmodel_wc_explainer_%s_%s.pkl'%(resultDir,pol,sitename), 'w+b') as fp: 
        pickle.dump(explainer, fp)    

    nrows = 1200
    shap_values = model_shap_values_mtp(x_explain,nrows,RFmodel_wc,number_of_processors=number_of_processors)
    with open(r'%s/shap_values_%s_%s.pkl'%(resultDir,pol,sitename), 'w+b') as fp: 
        pickle.dump(shap_values, fp)