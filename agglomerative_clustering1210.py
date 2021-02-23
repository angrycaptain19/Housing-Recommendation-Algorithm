#房源推荐算法——基于 Airbnb 北京房价预测的机器学习模型
#第14组
#by 何智钧

#Agglomerative Clustering
#2020-12-10

#导入第三方库
import pandas as pd
import re
import datetime
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler

#读取listing.csv并简要查看文件内容
listing_data=pd.read_csv("E:/钧资料3/大学资料/课程/LN3125数据挖掘与机器学习/final project/beijing1026/listings/listings.csv",encoding="ANSI",na_values="N/A")
print(listing_data)
print(listing_data.describe())
print(listing_data.info())
print(listing_data.dtypes)
print(listing_data.shape)
for label in listing_data.columns:
    print(label)
    print(listing_data[[label]].head(5))
    print("Missing values: {}".format(listing_data[[label]].isnull().sum()))
    print("\n\n")

#host_since转为到现在(2020/12/1)的时间
def to_today(date):
    present = datetime.datetime.strptime('2020-12-1',"%Y-%m-%d")
    if date<present:
        num=(present-date).days
        return num
    elif date>=present:
        num=(date-present).days
        return num
listing_data['host_since'] = pd.to_datetime(listing_data['host_since'])
listing_data['host_time'] = listing_data['host_since'].apply(lambda x:to_today(x))
listing_data['first_review'] = pd.to_datetime(listing_data['first_review'])
listing_data['first_review_to_today'] = listing_data['first_review'].apply(lambda x:to_today(x))
listing_data['last_review'] = pd.to_datetime(listing_data['last_review'])
listing_data['last_review_to_today'] = listing_data['last_review'].apply(lambda x:to_today(x))

#计算Bathroom的数量
def countBathroom(x):
    if type(x) is str:
        if (x=='Shared half-bath')|(x=='Private half-bath')|(x=='Half-bath'):
            num = 0.5
        else:
            num = float(re.findall(r'[\d+(\.\d+)?]',x)[0])
    else:
        num = 0
    return num
listing_data["bathrooms"] = listing_data.bathrooms_text.apply(lambda x: countBathroom(x))

#将百分数转化为数值
listing_data["host_response_rate"]=listing_data["host_response_rate"].str.strip("%").astype('float')/100
listing_data["host_acceptance_rate"]=listing_data["host_acceptance_rate"].str.strip("%").astype('float')/100

#将reviews_per_month的缺失值填充为0
listing_data['reviews_per_month'].fillna(0, inplace = True)

# 字符串列表转化为长度
def getLen(x):
    li = x[2:-2].split(',')
    length = len(li)
    return length
listing_data["num_of_host_verifications"] = listing_data.host_verifications.apply(lambda x:getLen(x))
listing_data["num_of_amenities"] = listing_data.amenities.apply(lambda x:getLen(x))

#price去掉美元符号，转为float
listing_data["price"]=listing_data["price"].str.replace(',', '').apply(lambda x : x[1:]).astype('float')

#生成虚拟变量
host_is_superhost_dummy_variables=pd.get_dummies(listing_data['host_is_superhost'],drop_first=False,prefix='host_is_superhost')
host_has_profile_pic_dummy_variables=pd.get_dummies(listing_data['host_has_profile_pic'],drop_first=False,prefix='host_has_profile_pic')
host_identity_verified_dummy_variables=pd.get_dummies(listing_data['host_identity_verified'],drop_first=False,prefix='host_identity_verified')
property_type_dummy_variables=pd.get_dummies(listing_data['property_type'],drop_first=False,prefix='property_type')
room_type_dummy_variables=pd.get_dummies(listing_data['room_type'],drop_first=False,prefix='room_type')
instant_bookable_dummy_variables=pd.get_dummies(listing_data['instant_bookable'],drop_first=False,prefix='instant_bookable')
neighbourhood_cleansed_dummy_variables=pd.get_dummies(listing_data['neighbourhood_cleansed'],drop_first=False,prefix='neighbourhood_cleansed')
host_response_time_dummy_variables=pd.get_dummies(listing_data['host_response_time'],drop_first=False,prefix='host_response_time')
listing_data=pd.concat([listing_data,host_is_superhost_dummy_variables,host_has_profile_pic_dummy_variables,host_identity_verified_dummy_variables,property_type_dummy_variables,room_type_dummy_variables,instant_bookable_dummy_variables,neighbourhood_cleansed_dummy_variables,host_response_time_dummy_variables],axis=1)

#定义自变量和因变量的名字
X_names=["host_time",
                #"first_review_to_today",
                #"last_review_to_today",
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
                #"review_scores_rating",
                #"review_scores_accuracy",
                #"review_scores_cleanliness",
                #"review_scores_checkin",
                #"review_scores_communication",
                #"review_scores_location",
                #"review_scores_value",
                "calculated_host_listings_count",
                "calculated_host_listings_count_entire_homes",
                "calculated_host_listings_count_private_rooms",
                "calculated_host_listings_count_shared_rooms",
                #"reviews_per_month",
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
                "neighbourhood_cleansed_延庆县 / Yanqing",]
Y_name=["price"]

#选择X和Y的数据，并去除有缺失值的样本
XY_data=listing_data[["id",
                "host_time",
                #"first_review_to_today",
                #"last_review_to_today",
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
                #"review_scores_rating",
                #"review_scores_accuracy",
                #"review_scores_cleanliness",
                #"review_scores_checkin",
                #"review_scores_communication",
                #"review_scores_location",
                #"review_scores_value",
                "calculated_host_listings_count",
                "calculated_host_listings_count_entire_homes",
                "calculated_host_listings_count_private_rooms",
                "calculated_host_listings_count_shared_rooms",
                #"reviews_per_month",
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
                "neighbourhood_cleansed_延庆县 / Yanqing",
                "price",]]
dropna_data=XY_data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
#导出聚类前完整的数据
dropna_data.to_csv("E:/钧资料3/大学资料/课程/LN3125数据挖掘与机器学习/final project/beijing1026/listings/listings1211.csv",
                 sep=',', na_rep='',columns=None, header=True,index=False,encoding="ANSI")

#定义自变量
X=dropna_data[["host_time",
                #"first_review_to_today",
                #"last_review_to_today",
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
                #"review_scores_rating",
                #"review_scores_accuracy",
                #"review_scores_cleanliness",
                #"review_scores_checkin",
                #"review_scores_communication",
                #"review_scores_location",
                #"review_scores_value",
                "calculated_host_listings_count",
                "calculated_host_listings_count_entire_homes",
                "calculated_host_listings_count_private_rooms",
                "calculated_host_listings_count_shared_rooms",
                #"reviews_per_month",
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
                "neighbourhood_cleansed_延庆县 / Yanqing",]].values
#定义标准化的自变量
#xscaler=StandardScaler(copy=True, with_mean=True, with_std=True)
#xscaler.fit(X)
#XStandardlized=xscaler.transform(X)

#定义因变量：首先去除千分位分隔符，然后去掉$号
#Y=dropna_data["price"].str.replace(',', '')
#Y=Y.apply(lambda x : x[1:]).astype('float')
#定义标准化的因变量
#yscaler=StandardScaler(copy=True, with_mean=True, with_std=True)
#yscaler.fit(pd.DataFrame(Y,columns=Y_name))
#YStandardlized=yscaler.transform(pd.DataFrame(Y,columns=Y_name))
#最后才转化为array
#Y=Y.values
#YStandardlized=YStandardlized.reshape(-1,1)

#读取刚才生成的新数据集
new_listing_data=pd.read_csv("E:/钧资料3/大学资料/课程/LN3125数据挖掘与机器学习/final project/beijing1026/listings/listings1211.csv",encoding="ANSI",na_values="N/A")
print(new_listing_data)
print(new_listing_data.describe())
print(new_listing_data.info())
print(new_listing_data.dtypes)
print(new_listing_data.shape)

#选择聚类用到的变量
cluster_data=new_listing_data.drop(["id"],axis=1,inplace=False)

#聚类前归一化
cluster_scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
cluster_data=cluster_scaler.fit_transform(cluster_data)
#聚类
cluster_model=AgglomerativeClustering(n_clusters=3, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None)
labels=cluster_model.fit_predict(cluster_data)
output_data=pd.concat([pd.DataFrame(labels,columns=["Label"]),new_listing_data],axis=1)
#导出聚类后的数据集
output_data.to_csv("E:/钧资料3/大学资料/课程/LN3125数据挖掘与机器学习/final project/beijing1026/listings/listings_clusters1211.csv",
                 sep=',', na_rep='',columns=None, header=True,index=False,encoding="ANSI")

