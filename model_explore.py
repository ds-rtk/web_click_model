# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 00:02:05 2019

@author: rkanjila
"""


'@XGboost
####################################

import xgboost as xgb
import lightgbm as lgb

model_xgb = xgb.XGBRegressor(
#                              colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.1, max_depth=5, 
#                              min_child_weight=1.7817, 
                             n_estimators=100,
#                              reg_alpha=0.4640, 
#                              reg_lambda=0.8571,
#                              subsample=0.5213, 
                             random_state =7, 
                             nthread = -1,verbosity=1)

model_xgb.fit(train_X,train_logY)

pred_df, rmse, r2, mape = log_model_evaluate(model_xgb,train_X,train_Y)
pred_df, rmse, r2, mape = log_model_evaluate(model_xgb,test_x,test_Y)

LGB

model_lgb = lgb.LGBMRegressor(objective='regression',
                              num_leaves=5,
                              learning_rate=0.1, n_estimators=100,
#                               max_bin = 55, bagging_fraction = 0.8,
#                               bagging_freq = 5, feature_fraction = 0.2319,
#                               feature_fraction_seed=9, bagging_seed=9,
#                               min_data_in_leaf =6, min_sum_hessian_in_leaf = 11
                             )

model_lgb.fit(train_X,train_logY)

pred_df, rmse, r2, mape = log_model_evaluate(model_lgb,train_X,train_Y)
pred_df, rmse, r2, mape = log_model_evaluate(model_lgb,test_x,test_Y)























