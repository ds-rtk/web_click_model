# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:49:46 2019

@author: rkanjila
"""
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor

from sklearn import metrics



#Model Evaluation code

def model_evaluate( model_,X,Y ):
    pred_df = pd.DataFrame( { "actual": Y, "predicted" : model_.predict( X ) } )
    pred_df['resid'] = pred_df.actual - pred_df.predicted
    rmse = np.sqrt( metrics.mean_squared_error( pred_df.actual, pred_df.predicted ) )
    r2 = metrics.explained_variance_score( pred_df.actual, pred_df.predicted)
    mae = metrics.mean_absolute_error(pred_df.actual, pred_df.predicted)  

    pred_df['ape'] = np.abs((pred_df.actual - pred_df.predicted)/pred_df.actual)
    mape = np.mean(pred_df['ape'])

    print( "RMSE: ", rmse, " : ", "R Squared: ", r2, "MAE: ", mae, "MAPE:", mape )
    return pred_df, rmse, r2, mape





gra_boost_m_ = GradientBoostingRegressor( learning_rate = 0.2, n_estimators = 200, max_depth = 5,verbose=2 )


gra_boost_m_.fit(train_X,train_Y)

pred_df, rmse, r2, mape = model_evaluate(gra_boost_m_,train_X,train_Y)
pred_df, rmse, r2, mape = model_evaluate(gra_boost_m_,test_x,test_Y)

pred_df = pd.DataFrame( { "actual": test_Y, "predicted" : gra_boost_m_.predict( test_x ) } )

pred_df.to_csv("test_pred.csv")



#gbm log


def log_model_evaluate( model_,X,Y ):
    
    pred_df = pd.DataFrame( { "actual": Y, "predicted" : np.exp(model_.predict( X )) } )
    pred_df['resid'] = pred_df.actual - pred_df.predicted
    rmse = np.sqrt( metrics.mean_squared_error( pred_df.actual, pred_df.predicted ) )
    r2 = metrics.explained_variance_score( pred_df.actual, pred_df.predicted)
    mae = metrics.mean_absolute_error(pred_df.actual, pred_df.predicted)  

    pred_df['ape'] = np.abs((pred_df.actual - pred_df.predicted)/pred_df.actual)
    mape = np.mean(pred_df['ape'])

    print( "RMSE: ", rmse, " : ", "R Squared: ", r2, "MAE: ", mae, "MAPE:", mape )
    return pred_df, rmse, r2, mape



gra_boost_l_ = GradientBoostingRegressor( learning_rate = 0.2, n_estimators = 100, max_depth = 5,verbose=2)

gra_boost_l_.fit(train_X,train_logY)

pred_df, rmse, r2, mape = model_evaluate(gra_boost_l_,train_X,train_logY)
pred_df, rmse, r2, mape = model_evaluate(gra_boost_l_,test_x,test_logY)

pred_df, rmse, r2, mape = log_model_evaluate(gra_boost_l_,train_X,train_Y)
pred_df, rmse, r2, mape = log_model_evaluate(gra_boost_l_,test_x,test_Y)


pred_df = pd.DataFrame( { "actual": test_Y, "predicted" : gra_boost_m_.predict( test_x ) , "predicted_log" : gra_boost_l_.predict( test_x )} )
pred_df.to_csv("test_pred.csv")

feat_imp = pd.Series(gra_boost_l_.feature_importances_,list(train_X.columns)).sort_values(ascending = False)
feat_imp.plot(kind="bar",title="feature importance")


