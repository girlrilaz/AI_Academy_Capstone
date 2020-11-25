import time,os,re,csv,sys,uuid,joblib
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
# from sklearn import ensemble
from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import classification_report

from logger import update_predict_log, update_train_log
from solution_guidance.cslib import fetch_data

import seaborn as sns
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
sns.set()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from functools import partial
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (TimeSeriesSplit, train_test_split, 
                                     cross_val_score)
from sklearn.preprocessing import StandardScaler, scale
from sklearn.metrics import make_scorer
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
import xgboost as xgb
from statsmodels.tsa.stattools import pacf, acf

import warnings
warnings.filterwarnings("ignore")


## model specific variables (iterate the version and note with each change)
if not os.path.exists(os.path.join(".", "models")):
    os.mkdir("models") 

MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "XGBoost on AAVAIL revenue"
SAVED_MODEL = os.path.join("models", "model-{}.joblib".format(re.sub("\.", "_", str(MODEL_VERSION))))

def load_data():
    
    DIR = os.getcwd()
    data_dir = os.path.join(".", "data")
    df = fetch_data(os.path.join(data_dir,'cs-train'))
    
    return df

def get_preprocessor(df):
    """
    return the preprocessing pipeline
    """

    ## preprocessing pipeline
    df['invoice'] = df['invoice'].str.replace(r"[a-zA-Z]",'').astype('int')
    
#     numeric_features = []
#     numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
#                                           ('scaler', StandardScaler())])

#     categorical_features = []
#     categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
#                                               ('onehot', OneHotEncoder(handle_unknown='ignore'))])

#     preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
#                                                    ('cat', categorical_transformer, categorical_features)])
    return df

def prepare_timeseries(df):
    """
    prepare time series data
    """
    
    # revenue time series data
    ts_rev = df[['invoice_date', 'country', 'price']].set_index('invoice_date')
    ts_rev_day = pd.DataFrame(ts_rev.groupby('country').resample('D')['price'].sum())
    #ts_rev_day = ts_rev.resample('D').sum() 
    ts_rev_day = ts_rev_day.reset_index()
    ts_rev_day.columns = ['country','ds', 'y']
    #ts_rev_day.columns = ['ds', 'y']

    # times_viewed time series data
    ts_tv = df[['invoice_date', 'country' ,'times_viewed']].set_index('invoice_date')
    ts_tv_day = pd.DataFrame(ts_tv.groupby('country').resample('D')['times_viewed'].sum())
    #ts_tv_day = ts_tv.resample('D').sum() 
    ts_tv_day = ts_tv_day.reset_index()
    ts_tv_day.columns = ['country','ds', 'y']
    #ts_tv_day.columns = ['ds', 'y']
    
    # prepare the data for the model
    ts1 = ts_rev_day.set_index('ds')
    ts1['y'] = ts1['y'].replace(0.00, ts1['y'].mean())  # replacing 0.00 values with mean in the ts data
    ts2 = ts_tv_day.set_index('ds')
    ts2['y'] = ts2['y'].replace(0.00, ts2['y'].mean())  # replacing 0.00 values with mean in the ts data
    ts3 = pd.concat([ts1,ts2], axis=1 )
    ts3.columns = ['country1','revenue', 'country1', 'times_viewed'] 
    #ts3.columns = ['revenue', 'times_viewed'] 
    ts3.columns = ts3.columns.str.strip()

    return ts1, ts2

# MAPE computation
def mape(y, yhat, perc=True):
    n = len(yhat.index) if type(yhat) == pd.Series else len(yhat)    
    mape = []
    for a, f in zip(y, yhat):
        # avoid division by 0
        if f > 1e-9:
            mape.append(np.abs((a - f)/a))
    mape = np.mean(np.array(mape))
    return mape * 100. if perc else mape

mape_scorer = make_scorer(mape, greater_is_better=False)

class TargetTransformer:
    """
    Perform some transformation on the time series
    data in order to make the model more performant and
    avoid non-stationary effects.
    """
        
    def __init__(self, log=False, detrend=False, diff=False):
        
        self.trf_log = log
        self.trf_detrend = detrend
        self.trend = pd.Series(dtype=np.float64)
    
    def transform(self, index, values):
        """
        Perform log transformation to the target time series

        :param index: the index for the resulting series
        :param values: the values of the initial series

        Return:
            transformed pd.Series
        """
        res = pd.Series(index=index, data=values)

        if self.trf_detrend:
            self.trend = TargetTransformer.get_trend(res) - np.mean(res.values)
            res = res.subtract(self.trend)
            
        if self.trf_log:
            res = pd.Series(index=index, data=np.log(res.values))
        
        return res
    
    def inverse(self, index, values):
        """
        Go back to the original time series values

        :param index: the index for the resulting series
        :param values: the values of series to be transformed back

        Return:
            inverse transformed pd.Series
        """        
        res = pd.Series(index=index, data=values)
        
        if self.trf_log:
            res = pd.Series(index=index, data=np.exp(values))
        try:
            if self.trf_detrend:
                assert len(res.index) == len(self.trend.index)                
                res = res + self.trend
                
        except AssertionError:
            print("Use a different transformer for each target to transform")
            
        return res
    
    @staticmethod
    def get_trend(data):
        """
        Get the linear trend on the data which makes the time
        series not stationary
        """
        n = len(data.index)
        X = np.reshape(np.arange(0, n), (n, 1))
        y = np.array(data)
        model = LinearRegression()
        model.fit(X, y)
        trend = model.predict(X)
        return pd.Series(index=data.index, data=trend)

## prepare features data (lags in time series as X)
def optimal_lag_features_threshold(target, lags=None):

    if lags is None:
        partial = pd.Series(data=pacf(target, nlags=48))
        partial_abs = np.abs(partial)
        opt_thres = np.abs(partial_abs[1:]).median()
        
    return opt_thres

def create_lag_features(target, lags=None, thres=0.2):

    scaler = StandardScaler()
    features = pd.DataFrame()
                
    if lags is None:
        partial = pd.Series(data=pacf(target, nlags=48))
        lags = list(partial[np.abs(partial) >= thres].index)

    df = pd.DataFrame()
    if 0 in lags:
        lags.remove(0) # do not consider itself as lag feature
    for l in lags:
        df[f"lag_{l}"] = target.shift(l)
        
    features = pd.DataFrame(scaler.fit_transform(df[df.columns]), 
                            columns=df.columns)

    features = df
    features.index = target.index
    
    return features

def create_ts_features(data):
    
    def get_shift(row):
        """
        Factory working shift: 3 shifts per day of 8 hours
        """
        if 6 <= row.hour <= 14:
            return 2
        elif 15 <= row.hour <= 22:
            return 3
        else:
            return 1
    
    features = pd.DataFrame()
    
    features["hour"] = data.index.hour
    features["weekday"] = data.index.week
    features["dayofyear"] = data.index.dayofyear
    features["is_weekend"] = data.index.weekday.isin([5, 6]).astype(np.int32)
    features["weekofyear"] = data.index.weekofyear
    features["month"] = data.index.month
    features["season"] = (data.index.month%12 + 3)//3
    features["shift"] = pd.Series(data.index.map(get_shift))
    
    features.index = data.index
        
    return features

def model_train(feature_data, country, test=False, forecast_days = 30, seasonal_period = 5):
    """
    example funtion to train model
    
    The 'test' flag when set to 'True':
        (1) subsets the data and serializes a test version
        (2) specifies that the use of the 'test' log file

    Note that the latest training data is always saved to be used by perfromance monitoring tools.
    """

    ## start timer for runtime
    time_start = time.time()
    
    ## prepare target data (y)
    # 1 month forecast
    FCAST_STEPS = forecast_days
    # seasonal period inferred from the autocorrelation function
    SEASONAL_PERIOD = seasonal_period
    
    # the complete time series
    if country != '' :
        model_data = feature_data[feature_data['country'] == country]
    elif country == '' :
        model_data = feature_data.resample('D').sum() 
    c_target = model_data["y"]


    # data used for training
    date = c_target.index[-1] - pd.Timedelta(hours=FCAST_STEPS)
    t_target = c_target[c_target.index <= date]

    # data used for forecasting
    f_target = c_target[c_target.index > date]
    fcast_initial_date = f_target.index[0]
    fcast_range = pd.date_range(fcast_initial_date, periods=FCAST_STEPS, freq="D")

    print(f"Full available time range: from {c_target.index[0]} to {c_target.index[-1]}")
    print(f"Training time range: from {t_target.index[0]} to {t_target.index[-1]}")
    print(f"Short forecasting time range: from {fcast_range[0]} to {fcast_range[-1]}")

    opt_thres = optimal_lag_features_threshold(t_target, lags=None)
    lags = create_lag_features(t_target, thres=opt_thres)
    ts = create_ts_features(t_target)
    features = ts.join(lags, how="outer").dropna()
    target = t_target[t_target.index >= features.index[0]]

    # ## subset the data to enable faster unittests
    if test:
        n_samples = int(np.round(0.9 * features.shape[0]))
        subset_indices = np.random.choice(np.arange(features.shape[0]), n_samples, replace=False).astype(int)
        mask = np.in1d(np.arange(target.size),subset_indices)
        target=target[mask]
        features=features[mask]  
    
    # ## Perform a train-test split and find optimal target
    X_train, X_test, y_train, y_test = train_test_split(features, 
                                                        target, 
                                                        test_size=0.3,
                                                        shuffle=False) 

    y_train_trf = TargetTransformer(log=False, detrend=False)
    y_train = y_train_trf.transform(y_train.index, y_train.values)

    y_test_trf = TargetTransformer(log=False, detrend=False)
    y_test = y_test_trf.transform(y_test.index, y_test.values)

    print("... grid searching")
    # run XGBoost algorithm with hyperparameters optimization
    # this model outperforms the linear regression

    def train_xgb(params, X_train, y_train):
        """
        Train XGBoost regressor using the parameters given as input. The model
        is validated using standard cross validation technique adapted for time series
        data. This function returns a friendly output for the hyperopt parameter optimization
        module.
        
        Parameters
        ----------
        params: dict with the parameters of the XGBoost regressor. For complete list see: 
                https://xgboost.readthedocs.io/en/latest/parameter.html
        X_train: pd.DataFrame with the training set features
        y_train: pd.Series with the training set targets    
        
        Returns
        -------
        dict with keys 'model' for the trained model, 'status' containing the hyperopt
        status string and 'loss' with the RMSE obtained from cross-validation
        """
        
        n_estimators = int(params["n_estimators"])
        max_depth= int(params["max_depth"])

        try:
            model = xgb.XGBRegressor(n_estimators=n_estimators, 
                                    max_depth=max_depth, 
                                    learning_rate=params["learning_rate"],
                                    subsample=params["subsample"])

            result = model.fit(X_train, 
                            y_train.values.ravel(),
                            eval_set=[(X_train, y_train.values.ravel())],
                            early_stopping_rounds=50,
                            verbose=False)
            
            # cross validate using the right iterator for time series
            cv_space = TimeSeriesSplit(n_splits=5)
            cv_score = cross_val_score(model, 
                                    X_train, y_train.values.ravel(), 
                                    cv=cv_space, 
                                    scoring=mape_scorer)

            rmse = np.abs(np.mean(np.array(cv_score)))
            return {
                "loss": rmse,
                "status": STATUS_OK,
                "model": model
            }
            
        except ValueError as ex:
            return {
                "error": ex,
                "status": STATUS_FAIL
            }
        
    def optimize_xgb(X_train, y_train, max_evals=10):
        """
        Run Bayesan optimization to find the optimal XGBoost algorithm
        hyperparameters.
        
        Parameters
        ----------
        X_train: pd.DataFrame with the training set features
        y_train: pd.Series with the training set targets
        max_evals: the maximum number of iterations in the Bayesian optimization method
        
        Returns
        -------
        best: dict with the best parameters obtained
        trials: a list of hyperopt Trials objects with the history of the optimization
        """
        
        space = {
            "n_estimators": hp.quniform("n_estimators", 100, 1000, 10),
            "max_depth": hp.quniform("max_depth", 1, 8, 1),
            "learning_rate": hp.loguniform("learning_rate", -5, 1),
            "subsample": hp.uniform("subsample", 0.8, 1),
            "gamma": hp.quniform("gamma", 0, 100, 1)
        }

        objective_fn = partial(train_xgb, 
                            X_train=X_train, 
                            y_train=y_train)
        
        trials = Trials()
        best = fmin(fn=objective_fn,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=max_evals,
                    trials=trials)

        # evaluate the best model on the test set
        print(f"""
        Best parameters:
            learning_rate: {best["learning_rate"]} 
            n_estimators: {best["n_estimators"]}
            max_depth: {best["max_depth"]}
            sub_sample: {best["subsample"]}
            gamma: {best["gamma"]}
        """)
        return best, trials

    # ## fit model on training data
    best, trials = optimize_xgb(X_train, y_train, max_evals=50)
    
    # evaluate the best model on the test set
    res = train_xgb(best, X_test, y_test)
    xgb_model = res["model"]
    predictions = xgb_model.predict(X_test)
    cv_score = min([f["loss"] for f in trials.results if f["status"] == STATUS_OK])
    test_score = mape(y_test.values, predictions)

    print(f"Root mean square error cross-validation/test: {cv_score:.4f} / {test_score:.4f}")

    if test:
        print("... saving test version of model")
        joblib.dump(xgb_model, os.path.join("models", "test.joblib"))
    else:
        print("... saving model: {}".format(SAVED_MODEL))
        joblib.dump(xgb_model, SAVED_MODEL)

        print("... saving latest data")
        data_file = os.path.join("models", 'latest-train.pickle')
        with open(data_file, 'wb') as tmp:
            pickle.dump({'y':target, 'X':features}, tmp)

    ## generate reports in csv and charts and save in report/images and report/predictions folders
    # inverse transform
    actual = y_test
    predictions = pd.Series(data=predictions, index=y_test.index)
    # predictions = transformer.inverse(y_test.index, predictions)
    # actual = transformer.inverse(y_test.index, y_test.values)

    # plot predictions on the test set
    fig4, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    _ = xgb.plot_importance(xgb_model, height=0.9, ax=ax1)
    ax2.plot(actual, label="target", color="blue")
    ax2.plot(predictions, label="predicted", color="red")
    plt.title(f"XGBoost Target vs Predicted. Test RMSE : {test_score}")
    ax2.legend()
    #plt.show()

    fig4.savefig('report/images/XGBoost_Result.png')

    # store the results
    xgb_predictions = predictions.copy()
    xgb_predictions = pd.DataFrame(xgb_predictions, columns = ['pred_price'])
    xgb_predictions.to_csv('report/predictions/XGBoost_Prediction_Result.csv')
    
    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    print(f"Model training completed in {runtime}")

    ## update the log file
    update_train_log(features.shape, test_score, runtime,
                     MODEL_VERSION, MODEL_VERSION_NOTE, test=test)

    return target, features

def model_predict(query, target, features, country, date_input, model=None, forecast_days = 30, test=False):
    """
    example funtion to predict from model
    """

    ## start timer for runtime
    time_start = time.time()

    # 1 month forecast
    FCAST_STEPS = forecast_days
    
    ## input checks
    if isinstance(query, dict):
        query = pd.DataFrame(query)
    elif isinstance(query, pd.DataFrame):
        pass
    else:
        raise Exception("ERROR (model_predict) - invalid input. {} was given".format(type(query)))

    ## load model if needed
    if not model:
        model = model_load()
    
    ## output checking
    if len(query.shape) == 1:
        query = query.reshape(1, -1)

    ## lags used in building the features for the one-step ahead model
    feature_lags = [int(f.split("_")[1]) for f in features if "lag" in f]


    ## target series used for forecasting
    fcast_trf = TargetTransformer(log=False, detrend=False)
    y = fcast_trf.transform(target.index, target.values) 

    def forecast_multi_recursive(y, model, lags, n_steps=FCAST_STEPS, step="1D"):
    
        """Multi-step recursive forecasting using the input time 
        series data and a pre-trained machine learning model
        
        Parameters
        ----------
        y: pd.Series holding the input time-series to forecast
        model: an already trained machine learning model implementing the scikit-learn interface
        lags: list of lags used for training the model
        n_steps: number of time periods in the forecasting horizon
        step: forecasting time period given as Pandas time series frequencies
        
        Returns
        -------
        fcast_values: pd.Series with forecasted values indexed by forecast horizon dates 
        """
        
        # get the dates to forecast
        last_date = y.index[-1] + pd.Timedelta(hours=1)
        fcast_range = pd.date_range(last_date, periods=n_steps, freq=step)

        fcasted_values = []
        target = y.copy()

        for date in fcast_range:

            new_point = fcasted_values[-1] if len(fcasted_values) > 0 else 0.0   
            target = target.append(pd.Series(index=[date], data=new_point))

            # forecast
            ts_features = create_ts_features(target)
            if len(lags) > 0:
                lags_features = create_lag_features(target, lags=lags)
                features = ts_features.join(lags_features, how="outer").dropna()
            else:
                features = ts_features
                
            predictions = model.predict(features)
            fcasted_values.append(predictions[-1])

        return pd.Series(index=fcast_range, data=fcasted_values)

    rec_fcast = forecast_multi_recursive(y, model, feature_lags)

    y = fcast_trf.inverse(index=y.index, values=y.values)
    rec_fcast = fcast_trf.inverse(index=rec_fcast.index, values=rec_fcast.values)

    # plot resulting forecast
    start_date = y.index[-1] - pd.Timedelta(days=5)
    in_sample = y[y.index >= start_date]

    fig5, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(in_sample, color="blue", label="Original")
    ax.plot(rec_fcast, color="red", label="Recursive forecast")
    ax.set_title(f"{FCAST_STEPS} days recursive forecast")
    ax.legend()
    #plt.show()

    fig5.savefig('report/images/Recursive_Forecast_Result.png')

    rec_fcast_df = pd.DataFrame(rec_fcast).reset_index()
    rec_fcast_df.columns = ['date', 'forecasted_price']
    rec_fcast_df.to_csv('report/predictions/XGBoost_Forecast_Result.csv')
    rec_fcast_df['date'] = pd.to_datetime(rec_fcast_df['date'])
    rec_fcast_df['date'] = rec_fcast_df['date'].dt.date
    rec_fcast_df['date'] = pd.to_datetime(rec_fcast_df['date'])
    rec_fcast_df = rec_fcast_df.set_index('date')
    rec_fcast_on_date = rec_fcast_df.loc[date_input][0]

    y_proba ='None'
    
    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update the log file
    for i in range(query.shape[0]):
        update_predict_log(rec_fcast_on_date, y_proba, query.iloc[i].values.tolist(), 
                           runtime, MODEL_VERSION, test=test)

    print(f"Model forecasting completed in {runtime}")
        
    return round(rec_fcast_on_date,2)
    


def model_load(test=False):
    """
    example funtion to load model
    """
    if test : 
        print( "... loading test version of model" )
        model = joblib.load(os.path.join("models","test.joblib"))
        return(model)

    if not os.path.exists(SAVED_MODEL):
        exc = "Model '{}' cannot be found did you train the full model?".format(SAVED_MODEL)
        raise Exception(exc)
    
    model = joblib.load(SAVED_MODEL)
    return(model)


if __name__ == "__main__":

    ## data ingestion
    df = load_data()

    df_processed = get_preprocessor(df)

    ts1, ts2 = prepare_timeseries(df_processed)

    # Input pre-selected prediction query i.e. country and date of prediction
    country = input ("Enter country name, i.e. Australia, to forecast all countries, leave blank and press Enter: ")  
    date_input = input("Enter data between 2019-07-30 to 2019-08-28 i.e. 2019-08-01 : ")
    date_input = datetime.strptime(date_input, '%Y-%m-%d')

    """
    basic test procedure for model.py
    """
    
    ## train the model
    target, features = model_train(ts1, country, test=False, forecast_days = 30, seasonal_period = 5)

    ## load the model
    model = model_load(test=False)
    
    ## example predict
    query = pd.DataFrame({'country': [country],
                          'date': [date_input]
    })

    result = model_predict(query, target, features, country, date_input, model, 30, test=True)
    print(f'The forecasted revenue for {country} on the {date_input} is : {result}')
