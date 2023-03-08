import numpy as np
from sklearn.decomposition import PCA  
from sklearn import preprocessing
data_pca = pd.read_csv(r'E:\工作资料\上海臭氧课题\空气质量数据\!!!带气象的站点数据\2013-2021清洗并填充后\SHdata_5sites_2013_2021_除超过6小时缺值天线性插值.csv',parse_dates=['datetime'])
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
# X = sklearn.datasets.load_iris().data
    data = pd.read_csv(r'E:\工作资料\上海臭氧课题\PCA\按10-17时0803\%s_all.csv'%sitename,parse_dates=['datetime'])
    # X = np.array(data[['datetime','sitename','O31','NOx','t2m','ws','wd']].iloc[:,2:])
    X = np.array(data[['datetime','sitename','O31','NOx','t2m','u10','v10']].iloc[:,2:])
    
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
    k1_spss_sign=k1_spss.T*k_sign #取正负号
    # k1_spss_sign
    print(k1_spss_sign)
    
    temp = -score_scaler[:,1]
    temp = pd.DataFrame(temp,index=data.datetime,columns=[sitename])
    rbgo3 = pd.concat([rbgo3,temp],axis=1)
rbgo3['fac2_mean'] = rbgo3.mean(axis=1)
rbgo3['BKGO3'] = rbgo3['fac2_mean']*np.mean(evrs)*np.std(data_pca['O31']) + np.mean(data_pca['O31'])
# rbgo3 = rbgo3.resample('M').mean()
# plt.plot(rbgo3.rbgo3)
rbgo3