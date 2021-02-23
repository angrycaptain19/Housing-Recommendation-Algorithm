#房源推荐算法——基于 Airbnb 北京房价预测的机器学习模型
#第14组
#by 卢嘉婷

#Multilayer Neural Network
#2020-12-07


#导入第三方库
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

#读取listing.csv并简要查看文件内容
cluster1=pd.read_excel('listings_clusters1211.xlsx',sheet_name='聚类1')
print(cluster1['host_time'][0])


#定义自变量和因变量的名字
X=cluster1[["host_time",
                "num_of_host_verifications",
                "num_of_amenities",
                "latitude",
                "longitude",
                "host_response_rate",
                "host_acceptance_rate",
                "host_listings_count",
                "accommodates",
                "bathrooms",
                "bedrooms",
                "beds",
                "minimum_nights",
                "maximum_nights",
                "availability_30",
                "availability_60",
                "availability_90",
                "availability_365",
                "number_of_reviews",
                "number_of_reviews_ltm",
#                "review_scores_rating",
#                "review_scores_accuracy",
#                "review_scores_cleanliness",
#                "review_scores_checkin",
#                "review_scores_communication",
#                "review_scores_location",
#                "review_scores_value",
                "calculated_host_listings_count",
                "calculated_host_listings_count_entire_homes",
                "calculated_host_listings_count_private_rooms",
                "calculated_host_listings_count_shared_rooms",
#                "reviews_per_month",
                "host_response_time_a few days or more",
                "host_response_time_within a day",
                "host_response_time_within a few hours",
                "host_response_time_within an hour",
                "host_is_superhost_t",
                "host_is_superhost_f",
                "host_has_profile_pic_t",
                "host_has_profile_pic_f",
                "host_identity_verified_t",
                "host_identity_verified_f",
                "property_type_Barn",
                "property_type_Camper/RV",
                "property_type_Campsite",
                "property_type_Casa particular",
                "property_type_Castle",
                "property_type_Cave",
                "property_type_Dome house",
                "property_type_Earth house",
                "property_type_Entire apartment",
                "property_type_Entire bed and breakfast",
                "property_type_Entire bungalow",
                "property_type_Entire cabin",
                "property_type_Entire chalet",
                "property_type_Entire condominium",
                "property_type_Entire cottage",
                "property_type_Entire guest suite",
                "property_type_Entire guesthouse",
                "property_type_Entire home/apt",
                "property_type_Entire house",
                "property_type_Entire loft",
                "property_type_Entire place",
                "property_type_Entire resort",
                "property_type_Entire serviced apartment",
                "property_type_Entire townhouse",
                "property_type_Entire villa",
                "property_type_Farm stay",
                "property_type_Houseboat",
                "property_type_Hut",
                "property_type_Igloo",
                "property_type_Kezhan",
                "property_type_Minsu",
                "property_type_Pension",
                "property_type_Private room",
                "property_type_Private room in apartment",
                "property_type_Private room in barn",
                "property_type_Private room in bed and breakfast",
                "property_type_Private room in bungalow",
                "property_type_Private room in cabin",
                "property_type_Private room in camper/rv",
                "property_type_Private room in campsite",
                "property_type_Private room in casa particular",
                "property_type_Private room in castle",
                "property_type_Private room in cave",
                "property_type_Private room in chalet",
                "property_type_Private room in condominium",
                "property_type_Private room in cottage",
                "property_type_Private room in dome house",
                "property_type_Private room in earth house",
                "property_type_Private room in farm stay",
                "property_type_Private room in guest suite",
                "property_type_Private room in guesthouse",
                "property_type_Private room in hostel",
                "property_type_Private room in house",
                "property_type_Private room in hut",
                "property_type_Private room in kezhan",
                "property_type_Private room in loft",
                "property_type_Private room in minsu",
                "property_type_Private room in nature lodge",
                "property_type_Private room in resort",
                "property_type_Private room in ryokan",
                "property_type_Private room in serviced apartment",
                "property_type_Private room in tent",
                "property_type_Private room in tiny house",
                "property_type_Private room in townhouse",
                "property_type_Private room in treehouse",
                "property_type_Private room in villa",
                "property_type_Room in aparthotel",
                "property_type_Room in boutique hotel",
                "property_type_Room in heritage hotel",
                "property_type_Room in hotel",
                "property_type_Shared room",
                "property_type_Shared room in apartment",
                "property_type_Shared room in bed and breakfast",
                "property_type_Shared room in boutique hotel",
                "property_type_Shared room in bungalow",
                "property_type_Shared room in condominium",
                "property_type_Shared room in cottage",
                "property_type_Shared room in earth house",
                "property_type_Shared room in farm stay",
                "property_type_Shared room in guest suite",
                "property_type_Shared room in guesthouse",
                "property_type_Shared room in hostel",
                "property_type_Shared room in house",
                "property_type_Shared room in hut",
                "property_type_Shared room in kezhan",
                "property_type_Shared room in loft",
                "property_type_Shared room in nature lodge",
                "property_type_Shared room in serviced apartment",
                "property_type_Shared room in tent",
                "property_type_Shared room in tiny house",
                "property_type_Shared room in townhouse",
                "property_type_Shared room in villa",
                "property_type_Tent",
                "property_type_Tiny house",
                "property_type_Treehouse",
                "property_type_Yurt",
                "room_type_Entire home/apt",
                "room_type_Private room",
                "room_type_Shared room",
                "instant_bookable_t",
                "instant_bookable_f",
                "neighbourhood_cleansed_昌平区",
                "neighbourhood_cleansed_朝阳区 / Chaoyang",
                "neighbourhood_cleansed_大兴区 / Daxing",
                "neighbourhood_cleansed_东城区",
                "neighbourhood_cleansed_房山区",
                "neighbourhood_cleansed_丰台区 / Fengtai",
                "neighbourhood_cleansed_海淀区",
                "neighbourhood_cleansed_怀柔区 / Huairou",
                "neighbourhood_cleansed_门头沟区 / Mentougou",
                "neighbourhood_cleansed_密云县 / Miyun",
                "neighbourhood_cleansed_平谷区 / Pinggu",
                "neighbourhood_cleansed_石景山区",
                "neighbourhood_cleansed_顺义区 / Shunyi",
                "neighbourhood_cleansed_通州区 / Tongzhou",
                "neighbourhood_cleansed_西城区",
                "neighbourhood_cleansed_延庆县 / Yanqing",]]
Y=cluster1["price"]


#定义标准化的自变量
xscaler=StandardScaler(copy=True, with_mean=True, with_std=True)
xscaler.fit(X)
XStandardlized=np.array(xscaler.transform(X))

#定义标准化的因变量
yscaler=StandardScaler(copy=True, with_mean=True, with_std=True)
yscaler.fit(pd.DataFrame(Y))
YStandardlized=np.array(yscaler.transform(pd.DataFrame(Y))).reshape(-1,1)
print(XStandardlized[0][0])
print(YStandardlized[0])


#划分训练集和验证集
x_train,x_valid,y_train,y_valid=train_test_split(X,Y,
                                                 test_size=0.3,random_state=1210)
x_train_std,x_valid_std,y_train_std,y_valid_std=train_test_split(XStandardlized,YStandardlized,
                                                 test_size=0.3,random_state=1210)

#建立cluster1 的 multilayer neural network (unstandardized)
regr=MLPRegressor(hidden_layer_sizes=(80,),activation='identity',
                  solver='adam',alpha=0.00001,batch_size='auto',
                  learning_rate_init=0.001,learning_rate='constant',
                  max_iter=3000,shuffle=True,random_state=1210,
                  tol=1e-4,verbose=2,warm_start=False,early_stopping=True,
                  beta_1=0.9,beta_2=0.999,epsilon=1e-8,
                  n_iter_no_change=10)      
NN=regr.fit(x_train,y_train)
print('Cluster 1 Multilayer Neural Network (unstandardized)')
print('Train Score:  {}'.format(str(NN.score(x_train,y_train))))
print('Valid R^2:  {}'.format(str(r2_score(y_valid,NN.predict(x_valid)))))
print('Params:  {}'.format(str(NN.get_params)))




#建立cluster1 的 multilayer neural network (standardized)
regr_std=MLPRegressor(hidden_layer_sizes=(1500,),activation='identity',
                  solver='adam',alpha=0.00001,batch_size='auto',
                  learning_rate_init=0.0005,learning_rate='adaptive',
                  max_iter=5000,shuffle=True,random_state=None,
                  tol=1e-6,verbose=True,warm_start=False,early_stopping=False,
                  beta_1=0.9,beta_2=0.999,epsilon=1e-8,
                  n_iter_no_change=10)  
NN_std=regr_std.fit(x_train_std,y_train_std)
print('Cluster 1 Multilayer Neural Network (standardized)')
print('Train Score:')
print(NN_std.score(x_train_std,y_train_std))
print('Valid R^2:')
print(r2_score(y_valid_std,NN_std.predict(x_valid_std)))
print('Params:')
print(NN_std.get_params)
print('\n\n')






#对cluster1 的 multilayer neural network (standardized)调参建模
params_test1={
    'hidden_layer_sizes':[(1200,),(1500,),(1800,)],
    'activation':['identity','logistic'],
    'learning_rate':['constant','invscaling’','adaptive'],
    'solver':['lbfgs','sgd','adam'],
    }
regr_std_tailor=MLPRegressor(alpha=0.00001,batch_size='auto',learning_rate_init=0.0005,
                             max_iter=5000,shuffle=True,tol=1e-6,
                        verbose=True,warm_start=False)
gs1=GridSearchCV(regr_std_tailor,param_grid=params_test1,scoring=None,n_jobs=6,cv=5,verbose=2)
gs1.fit(XStandardlized,YStandardlized)
print('Cluster 1 Multilayer Neural Network (standardized)')
print('NN_std_tailor Best Score:')
print(gs1.best_score_)
print('NN_std_tailor Best Params:')
print(gs1.best_params_)
print('\n\n')





#对cluster1建立随机森林模型
regr_RF=RandomForestRegressor(n_estimators=900,criterion='mae',
                         max_depth=13,min_samples_split=6,
                         min_samples_leaf=1,min_weight_fraction_leaf=0.0,
                         max_features='auto',max_leaf_nodes=None,min_impurity_decrease=0.0,
                         min_impurity_split=None,bootstrap=False,n_jobs=6,verbose=2,warm_start=False)
RF=regr_RF.fit(x_train,y_train)
print('Cluster 1 Random Forest')
print('Train Score:  {}'.format(str(RF.score(x_train,y_train))))
print('Valid R^2:  {}'.format(str(r2_score(y_valid,RF.predict(x_valid)))))
print('Params:  {}'.format(str(RF.get_params)))
print('\n\n')




#对cluster1 的 随机森林调参建模
params_test2={
    'n_estimators':[500,600,700,800],
    'max_depth':[16,18,20,22,24,26],
    'max_features':[21,23,25,27],
    ‘criterion’:['mse','mae'],
    }
regr_RF_tailor=RandomForestRegressor(min_samples_leaf=2,min_samples_split=2,
                                     min_weight_fraction_leaf=0.0,
                                     min_impurity_decrease=0.0,min_impurity_split=None,bootstrap=True,
                                     oob_score=True,n_jobs=6,verbose=2,warm_start=False)
gs2=GridSearchCV(regr_RF_tailor,param_grid=params_test2,scoring=None,n_jobs=6,cv=5,verbose=2)
gs2.fit(X,Y)
best_score=gs2.best_score_
best_params=gs2.best_params_
print('Cluster 1')
print('RF_tailor Best Score:  {}'.format(str(best_score)))
print('RF_tailor Best Params:  {}'.format(str(best_params)))
feature_importances_CV=gs2.best_estimator_.feature_importances_
feature_importances_CV_list=list(zip(cluster1.columns[2:],feature_importances_CV))
feature_importances_CV_list.sort(key=lambda x:x[1],reverse=True)
feature_importances_CV_sort=dict(feature_importances_CV_list)
print('RF_tailor Features Importance:')
print(feature_importances_CV_sort)
print('\n\n')




