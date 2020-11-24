# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 17:13:04 2019

@author: rkanjila
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

#Data loading

data = pd.read_csv("C:/reet_personal/trivago/sem_bidding_cs.csv")

modelDF = data[data["week"] != 20190701]
submitDF = data[data["week"] == 20190701]
print ("Model data shape", modelDF.shape)
print ("Submission data shape", submitDF.shape)



#NA check



def na_check(df):
    nd_df = df.isna().sum()
    nd_vars = nd_df[nd_df>0]
    print ("Variable with nulls:\n",nd_vars)
    
na_check(modelDF)
na_check(submitDF)

# removing missing bids from the data

modelDF1 = modelDF[modelDF["bid"].isna()==False]
submitDF1 = submitDF[submitDF["bid"].isna()==False]

print ("Model data shape", modelDF1.shape)
print ("Submission data shape", submitDF1.shape)


#Feature Engg - Cat

def cat_feat(df):
    df1 = df.copy()
    dummy_cat = pd.get_dummies(df1[["locale","device","theme","match_type"]],
                               columns=["locale","device","theme","match_type"],dtype=int)
    
    df2 = pd.concat([df1,dummy_cat],axis = 1)
#    df2.drop(["Unnamed: 0","search_id","locale","theme","match_type"],axis = 1,inplace=True)
    print ("Input Data Shape:", df.shape)
    print ("Output Data Shape:", df2.shape)
    return(df2)

modelDF2 = cat_feat(modelDF1)
submitDF2 = cat_feat(submitDF1)


def poly_interaction(df1):
    
    df1["num_impressions*locale_AU"] = df1["num_impressions"]*df1["locale_AU"]
    df1["num_impressions*locale_BR"] = df1["num_impressions"]*df1["locale_BR"]
    df1["num_impressions*locale_DE"] = df1["num_impressions"]*df1["locale_DE"]
    df1["num_impressions*locale_ES"] = df1["num_impressions"]*df1["locale_ES"]
    df1["num_impressions*locale_FR"] = df1["num_impressions"]*df1["locale_FR"]
    df1["num_impressions*locale_IT"] = df1["num_impressions"]*df1["locale_IT"]
    df1["num_impressions*locale_UK"] = df1["num_impressions"]*df1["locale_UK"]
    
    df1["num_impressions*device_phone"] = df1["num_impressions"]*df1["device_phone"]
    df1["num_impressions*device_tablet"] = df1["num_impressions"]*df1["device_tablet"]
    df1["num_impressions*device_desktop"] = df1["num_impressions"]*df1["device_desktop"]
    
    df1["num_impressions*theme_CityOnly"] = df1["num_impressions"]*df1["theme_CityOnly"]
    df1["num_impressions*theme_Country"] = df1["num_impressions"]*df1["theme_Country"]
    df1["num_impressions*theme_HT"] = df1["num_impressions"]*df1["theme_HT"]
    df1["num_impressions*theme_Item"] = df1["num_impressions"]*df1["theme_Item"]
    df1["num_impressions*theme_POI"] = df1["num_impressions"]*df1["theme_POI"]
    df1["num_impressions*theme_POIOnly"] = df1["num_impressions"]*df1["theme_POIOnly"]
    df1["num_impressions*theme_Region"] = df1["num_impressions"]*df1["theme_Region"]
    df1["num_impressions*theme_RegionOnly"] = df1["num_impressions"]*df1["theme_RegionOnly"]
    
    df1["bid*locale_AU"] = df1["bid"]*df1["locale_AU"]
    df1["bid*locale_BR"] = df1["bid"]*df1["locale_BR"]
    df1["bid*locale_DE"] = df1["bid"]*df1["locale_DE"]
    df1["bid*locale_ES"] = df1["bid"]*df1["locale_ES"]
    df1["bid*locale_FR"] = df1["bid"]*df1["locale_FR"]
    df1["bid*locale_IT"] = df1["bid"]*df1["locale_IT"]
    df1["bid*locale_UK"] = df1["bid"]*df1["locale_UK"]
    
    df1["bid*device_phone"] = df1["bid"]*df1["device_phone"]
    df1["bid*device_tablet"] = df1["bid"]*df1["device_tablet"]
    df1["bid*device_desktop"] = df1["bid"]*df1["device_desktop"]
    
    df1["bid*theme_CityOnly"] = df1["bid"]*df1["theme_CityOnly"]
    df1["bid*theme_Country"] = df1["bid"]*df1["theme_Country"]
    df1["bid*theme_HT"] = df1["bid"]*df1["theme_HT"]
    df1["bid*theme_Item"] = df1["bid"]*df1["theme_Item"]
    df1["bid*theme_POI"] = df1["bid"]*df1["theme_POI"]
    df1["bid*theme_POIOnly"] = df1["bid"]*df1["theme_POIOnly"]
    df1["bid*theme_Region"] = df1["bid"]*df1["theme_Region"]
    df1["bid*theme_RegionOnly"] = df1["bid"]*df1["theme_RegionOnly"]
    

poly_interaction(modelDF2)
poly_interaction(submitDF2)
    


#Feature Engg - Num


def cont_feat(df1):
#    df1 = df.copy()
    df1["log_nup_imprs"] = np.log(df1["num_impressions"])
  
    df1["top3placement"] = df1["ad_position"].apply(lambda x: 1 if x<=3 else 0)
    df1["top1placement"] = df1["ad_position"].apply(lambda x: 1 if x<=1 else 0)
    
    df1["quality_score_hi"] = df1["quality_score"].apply(lambda x: 1 if x >= 6 else 0)
    df1["quality_score_sqr"] = df1["quality_score"]**2
    
    df1["bid_great"] = df1["bid"].apply(lambda x: 1 if x >= 10.0 else 0)
    df1["bid_good"] = df1["bid"].apply(lambda x: 1 if (x >= 5 and x < 10) else 0)
    
    df1["quality_score*ad_pos"] = df1["quality_score"]*df1["ad_position"]
#    df1["quality_score_by_ad_pos"] = df1["quality_score"]/df1["ad_position"]
#    df1["quality_score_by_ad_pos"].fillna(0,inplace=True)
#    df1["quality_score_by_ad_pos"] = df1["quality_score_by_ad_pos"].replace(np.inf,0)
#    df1["quality_score_by_ad_pos_sqrt"] = df1["quality_score"]/np.sqrt(df1["ad_position"])
#    df1["quality_score_by_ad_pos_sqrt"].fillna(0,inplace=True)
#    df1["quality_score_by_ad_pos_sqrt"] = df1["quality_score_by_ad_pos_sqrt"].replace(np.inf,0)
    df1["bid*ad_pos"] = df1["bid"]*df1["ad_position"]
    df1["bid*quality_score"] = df1["bid"]*df1["quality_score"]
    
    df1["bid*quality_score*ad_pos"] = df1["bid"]*df1["quality_score"]*df1["ad_position"]
    
    
    
#    return(df1)


cont_feat(modelDF2)
cont_feat(submitDF2)




#check

def na_check(df):
    nd_df = df.isna().sum()
    nd_vars = nd_df[nd_df>0]
    inf_var = df.columns[np.isfinite(df).all() == False]
    print ("Variable with nulls:\n",nd_vars,inf_var)
    
na_check(modelDF2)
na_check(submitDF2)



#Train test split

trainDF = modelDF2[modelDF2["week"] < 20181224]
testDF = modelDF2[modelDF2["week"] >= 20181224]
print (trainDF["week"].nunique())
print (testDF["week"].nunique())

print ("Train data shape", trainDF.shape)
print ("Test data shape", testDF.shape)


#Create X,Y


def create_X(df):
    print ("Input data shape", df.shape)
    X = df.drop(["week","Unnamed: 0","search_id","device","locale","theme","match_type","theme_City","locale_UK","match_type_EM","device_tablet"],\
                axis=1,inplace=False)
    X.drop("clicks",axis=1,inplace=True)
    print ("Output data shape", X.shape)
    return X

def create_Y(df):
    Y = df["clicks"]
    return Y




train_X = create_X(trainDF)
test_x = create_X(testDF)

train_Y = create_Y(trainDF)
test_Y = create_Y(testDF)



def create_logY(df):
    Y = np.log(df["clicks"])
    
    return Y

train_logY = create_logY(trainDF)
test_logY = create_logY(testDF)


na_check(train_Y)
na_check(train_logY)


np.isfinite(train_X).all()