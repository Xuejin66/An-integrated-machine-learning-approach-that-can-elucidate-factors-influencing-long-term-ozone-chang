import numpy as np
import pandas as pd
import math
from sklearn.decomposition import PCA  
from sklearn import preprocessing

data_pca = pd.read_csv(r'E:\Desktop\Ozone in Shanghai\An-integrated-machine-learning-approach-that-can-elucidate-factors-influencing-long-term-ozone-chang\inputData\airQuality&metData.csv',parse_dates=['datetime'])
data_pca = data_pca.loc[(data_pca['datetime'].dt.hour>9)&(data_pca['datetime'].dt.hour<18)]
data_pca = data_pca.loc[(data_pca['datetime'].dt.month>3)&(data_pca['datetime'].dt.month<10)]

data_pca['wd'] = data_pca.apply(lambda x:(270-(180/np.pi)*math.atan2(x.v10,x.u10))%360,axis=1)
data_pca['ws'] = data_pca.apply(lambda x:np.sqrt((x.u10)**2+(x.v10)**2),axis=1)
data_pca['t2m'] = data_pca['t2m'] - 273.15
data_pca['NOx'] = data_pca['NO'] + data_pca['NO2']
data_pca = data_pca[['datetime','sitename','O31','NOx','t2m','ws','wd']]
data_pca = data_pca.loc[data_pca['sitename'].isin(['PT','DSL','PDHN'])]

rbgo3 = pd.DataFrame()
evrs = []
for sitename in ['PT','DSL','PDHN']:
    data = pd.read_csv(r'E:\Desktop\Ozone in Shanghai\An-integrated-machine-learning-approach-that-can-elucidate-factors-influencing-long-term-ozone-chang\inputData\%s_all.csv'%sitename,parse_dates=['datetime'])
    X = np.array(data[['datetime','sitename','O31','NOx','t2m','ws','wd']].iloc[:,2:])
    X_scaler = X.copy()

    scaler = preprocessing.StandardScaler().fit(X_scaler)
    X_scaler = scaler.transform(X_scaler,copy=False)

    pca = PCA(n_components=2).fit(X_scaler)
    score = pca.transform(X_scaler)
    scaler_score = preprocessing.StandardScaler().fit(score)
    score_scaler = scaler_score.transform(score)

    evr = pca.explained_variance_ratio_[1]/(pca.explained_variance_ratio_[1]+pca.explained_variance_ratio_[0])
    evrs.append(evr)
    
    k1_spss=pca.components_/np.sqrt(pca.explained_variance_.reshape(2,1))
    k_sign=np.sign(k1_spss.sum(axis=1))
    k1_spss_sign=k1_spss.T*k_sign 
    
    temp = -score_scaler[:,1]
    temp = pd.DataFrame(temp,index=data.datetime,columns=[sitename])
    rbgo3 = pd.concat([rbgo3,temp],axis=1)
rbgo3['fac2_mean'] = rbgo3.mean(axis=1)
rbgo3['BKGO3'] = rbgo3['fac2_mean']*np.mean(evrs)*np.std(data_pca['O31']) + np.mean(data_pca['O31'])
rbgo3.to_csv(r'E:\Desktop\Ozone in Shanghai\An-integrated-machine-learning-approach-that-can-elucidate-factors-influencing-long-term-ozone-chang\outputData\RBGO3.csv')