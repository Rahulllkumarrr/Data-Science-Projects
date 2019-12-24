# -*- coding: utf-8 -*-

# IMPORTING LIBRARIES

import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# DATA PREPERATION

## BRENT OIL FUTURES

### Brent Oil Futures Historical Data

brent_oil = pd.read_csv('Brent Oil Futures Historical Data.csv')

brent_oil.head()

### Dataset analysis

brent_oil[brent_oil.isnull().any(axis=1)].head()

import numpy as np
np.sum(brent_oil.isnull().any(axis=1))

brent_oil.isnull().any(axis=0)

brent_oil.info()

brent_oil.describe()

brent_oil=brent_oil.rename(columns={'Change %':'Change_oil'})
brent_oil=brent_oil[['Date','Change_oil']]

brent_oil.head()

## DAX

### DAX Historical Data


DAX= pd.read_csv('DAX Historical Data.csv')

DAX.head()

### Dataset analysis


DAX[DAX.isnull().any(axis=1)].head()

import numpy as np
np.sum(DAX.isnull().any(axis=1))

DAX.isnull().any(axis=0)

DAX.info()

DAX.describe()

### Getting change feature

DAX=DAX.rename(columns={'Change %':'Change_DAX'})
DAX=DAX[['Date','Change_DAX']]

DAX.head()

### Merging The data for change feature

aa=display('change_feature', 'DAX', "pd.merge(brent_oil, brent_oil, on='Date')")

change_feature =pd.merge(brent_oil, DAX, on='Date')

change_feature.head()

## GOLD FUTURES

### Gold Futures Historical Data


gold_features = pd.read_csv('Gold Futures Historical Data.csv')

gold_features.head()

### Dataset analysis


gold_features[gold_features.isnull().any(axis=1)].head()

import numpy as np
np.sum(gold_features.isnull().any(axis=1))

gold_features.isnull().any(axis=0)

gold_features.info()

gold_features.describe()

### Getting change feature

gold_features=gold_features.rename(columns={'Change %':'Change_gold'})
gold_features=gold_features[['Date','Change_gold']]

gold_features.head()

### Merging The data for change feature

aa=display('change_feature', 'gold_features', "pd.merge(brent_oil, brent_oil, on='Date')")

change_feature =pd.merge(change_feature, gold_features, on='Date')

change_feature.head()

## NIKKEI 225

### Nikkei 225 Historical Data

Nikkei= pd.read_csv('Nikkei 225 Historical Data.csv')

Nikkei.head()

### Dataset analysis

Nikkei[Nikkei.isnull().any(axis=1)].head()

import numpy as np
np.sum(Nikkei.isnull().any(axis=1))

Nikkei.isnull().any(axis=0)

Nikkei.info()

Nikkei.describe()

### Getting change feature

Nikkei=Nikkei.rename(columns={'Change %':'Change_Nikkei'})
Nikkei=Nikkei[['Date','Change_Nikkei']]

Nikkei.head()

### Merging The data for change feature

aa=display('change_feature', 'Nikkei', "pd.merge(change_feature, change_feature, on='Date')")

change_feature =pd.merge(change_feature, Nikkei, on='Date')

change_feature.head()

## S&P 500

### S&P 500 Historical Data

SP= pd.read_csv('S&P 500 Historical Data.csv')

SP.head()

### Dataset analysis

SP[SP.isnull().any(axis=1)].head()

import numpy as np
np.sum(SP.isnull().any(axis=1))

SP.isnull().any(axis=0)

SP.info()

SP.describe()

### Getting change feature

SP=SP.rename(columns={'Change %':'Change_SP'})
SP=SP[['Date','Change_SP']]

SP.head()

### Merging The data for change feature

aa=display('change_feature', 'SP', "pd.merge(change_feature, change_feature, on='Date')")

change_feature =pd.merge(change_feature, SP, on='Date')

change_feature.head()

## SHANGHAI COMPOSITE

### Shanghai Composite Historical Data

Shanghai= pd.read_csv('Shanghai Composite Historical Data.csv')

Shanghai.head()

### Dataset analysis

Shanghai [Shanghai .isnull().any(axis=1)].head()

import numpy as np
np.sum(Shanghai .isnull().any(axis=1))

Shanghai .isnull().any(axis=0)

Shanghai .info()

Shanghai .describe()

### Getting change feature

Shanghai=Shanghai.rename(columns={'Change %':'Change_Shanghai'})
Shanghai=Shanghai[['Date','Change_Shanghai']]

Shanghai.head()

### Merging The data for change feature

aa=display('change_feature', 'Shanghai', "pd.merge(change_feature, change_feature, on='Date')")

change_feature =pd.merge(change_feature, Shanghai, on='Date')

change_feature.head()

## USD-EUR

### USD_EUR Historical Data

USD_EUR= pd.read_csv('USD_EUR Historical Data.csv')

USD_EUR.head()

### Dataset analysis

USD_EUR [USD_EUR .isnull().any(axis=1)].head()

import numpy as np
np.sum(USD_EUR .isnull().any(axis=1))

USD_EUR .isnull().any(axis=0)

USD_EUR .info()

USD_EUR .describe()

### Getting change feature

USD_EUR=USD_EUR.rename(columns={'Change %':'Change_USD_EUR'})
USD_EUR=USD_EUR[['Date','Change_USD_EUR']]

USD_EUR.head()

### Merging The data for change feature

aa=display('change_feature', 'USD_EUR', "pd.merge(change_feature, change_feature, on='Date')")

change_feature =pd.merge(change_feature, USD_EUR, on='Date')

change_feature.head()

## USD-GBP

### USD_GBP Historical Data

USD_GBP= pd.read_csv('USD_GBP Historical Data.csv')

USD_GBP.head()

### Dataset analysis

USD_GBP [USD_GBP .isnull().any(axis=1)].head()

import numpy as np
np.sum(USD_GBP
       .isnull().any(axis=1))

USD_GBP .isnull().any(axis=0)

USD_GBP .info()

USD_GBP .describe()

### Getting change feature

USD_GBP=USD_GBP.rename(columns={'Change %':'Change_USD_GBP'})
USD_GBP=USD_GBP[['Date','Change_USD_GBP']]

USD_GBP.head()

### Merging The data for change feature

aa=display('change_feature', 'USD_GBP', "pd.merge(change_feature, change_feature, on='Date')")

change_feature =pd.merge(change_feature, USD_GBP, on='Date')

change_feature.head()

## USD-JPY

### USD_JPY Historical Data

USD_JPY= pd.read_csv('USD_JPY Historical Data.csv')

USD_JPY.head()

### Dataset analysis

USD_JPY [USD_JPY .isnull().any(axis=1)].head()

import numpy as np
np.sum(USD_JPY .isnull().any(axis=1))

USD_JPY.isnull().any(axis=0)

USD_JPY .info()

USD_JPY .describe()

### Getting change feature

USD_JPY=USD_JPY.rename(columns={'Change %':'Change_USD_JPY'})
USD_JPY=USD_JPY[['Date','Change_USD_JPY']]

USD_JPY.head()

### Merging The data for change feature

aa=display('change_feature', 'USD_JPY', "pd.merge(change_feature, change_feature, on='Date')")

change_feature =pd.merge(change_feature, USD_JPY, on='Date')

change_feature

change_feature = change_feature.dropna(axis = 0, how ='any')

change_feature

change_feature.to_csv('change_feature.csv')

change_feature.head()

import pandas as pd
change_feature  = pd.read_csv('change_feature.csv', index_col='Date', parse_dates=['Date'])
change_feature .head()

change_feature=change_feature.drop(columns=['Unnamed: 0'])
change_feature=change_feature.rename(columns={'Date':'date'})

aa=change_feature.iloc[::-1]
aa.to_csv('change_feature.csv')

change_feature  = pd.read_csv('change_feature.csv')
change_feature=change_feature.rename(columns={'Date':'date'})
change_feature.head()

# DATA CLEANING

## Cleaning % from all coloumns of change_feature

for letter in '%':
    change_feature['Change_oil']= change_feature['Change_oil'].str.replace(letter,'')

for letter in '%':
    change_feature['Change_DAX']= change_feature['Change_DAX'].str.replace(letter,'')

for letter in '%':
    change_feature['Change_gold']= change_feature['Change_gold'].str.replace(letter,'')

for letter in '%':
    change_feature['Change_Nikkei']= change_feature['Change_Nikkei'].str.replace(letter,'')

for letter in '%':
    change_feature['Change_SP']= change_feature['Change_SP'].str.replace(letter,'')

for letter in '%':
    change_feature['Change_Shanghai']= change_feature['Change_Shanghai'].str.replace(letter,'')

for letter in '%':
    change_feature['Change_USD_EUR']= change_feature['Change_USD_EUR'].str.replace(letter,'')

for letter in '%':
    change_feature['Change_USD_GBP']= change_feature['Change_USD_GBP'].str.replace(letter,'')

for letter in '%':
    change_feature['Change_USD_JPY']= change_feature['Change_USD_JPY'].str.replace(letter,'')

change_feature

# API FUNCTION DEFINING

def get_data(symbol):

    # API KEY: 51N1JMO66W56AYA3
    ti = TechIndicators(key='51N1JMO66W56AYA3', output_format='pandas')
    sma, _ = ti.get_sma(symbol=symbol, interval='daily')
    wma, _ = ti.get_wma(symbol=symbol, interval='daily')
    ema, _ = ti.get_ema(symbol=symbol, interval='daily')
    macd, _ = ti.get_macd(symbol=symbol, interval='daily')
    stoch, _ = ti.get_stoch(symbol=symbol, interval='daily')
    #  Alpha Vantage Times out for more than 5 request 
    time.sleep(65)
    rsi, _ = ti.get_rsi(symbol=symbol, interval='daily')
    adx, _ = ti.get_adx(symbol=symbol, interval='daily')
    cci, _ = ti.get_cci(symbol=symbol, interval='daily')
    aroon, _ = ti.get_aroon(symbol=symbol, interval='daily')
    bbands, _ = ti.get_bbands(symbol=symbol, interval='daily')
    time.sleep(65)
    ad, _ = ti.get_ad(symbol=symbol, interval='daily')
    obv, _ = ti.get_obv(symbol=symbol, interval='daily')
    mom, _ = ti.get_mom(symbol=symbol, interval='daily')
    willr, _ = ti.get_willr(symbol=symbol, interval='daily')
    time.sleep(65)
    tech_ind = pd.concat([sma, ema, macd,  rsi, adx, cci, aroon, bbands, ad, obv, wma, mom, willr, stoch], axis=1)

    ts = TimeSeries(key='51N1JMO66W56AYA3', output_format='pandas')
    close2 = ts.get_daily(symbol=symbol, outputsize='full')
    close = ts.get_daily(symbol=symbol, outputsize='full')[0]['4. close']
    direction = (close > close.shift()).astype(int)
    target = direction.shift(-1).fillna(0).astype(int)
    target.name = 'target'

    data = pd.concat([tech_ind, close, target], axis=1)

    return data

def get_indicators(data, n):

    hh = data['2. high'].rolling(n).max()
    ll = data['3. low'].rolling(n).min()
    up, dw = data['4. close'].diff(), -data['4. close'].diff()
    up[up<0], dw[dw<0] = 0, 0
    macd = data['4. close'].ewm(12).mean() - data['4. close'].ewm(26).mean()
    macd_signal = macd.ewm(9).mean()
    tp = (data['2. high'] + data['3. low'] + data['4. close']) / 3
    tp_ma = tp.rolling(n).mean()
    indicators = pd.DataFrame(data=0, index=data.index,
                              columns=['sma', 'ema', 'momentum',
                                       'sto_k', 'sto_d', 'rsi',
                                       'macd', 'lw_r', 'a/d', 'cci'])
    indicators['sma'] = data['4. close'].rolling(10).mean()
    indicators['ema'] = data['4. close'].ewm(10).mean()
    indicators['momentum'] = data['4. close'] - data['4. close'].shift(n)
    indicators['sto_k'] = (data['4. close'] - ll) / (hh - ll) * 100
    indicators['sto_d'] = indicators['sto_k'].rolling(n).mean()
    indicators['rsi'] = 100 - 100 / (1 + up.rolling(14).mean() / dw.rolling(14).mean())
    indicators['macd'] = macd - macd_signal
    indicators['lw_r'] = (hh - data['4. close']) / (hh - ll) * 100
    indicators['a/d'] = (data['2. high'] - data['4. close'].shift()) / (data['2. high'] - data['3. low'])
    indicators['cci'] = (tp - tp_ma) / (0.015 * tp.rolling(n).apply(lambda x: np.std(x)))

    return indicators

def rebalance(unbalanced_data):

    # Separate majority and minority classes
    data_minority = unbalanced_data[unbalanced_data.target==0]
    data_majority = unbalanced_data[unbalanced_data.target==1]

    # Upsample minority class
    n_samples = len(data_majority)
    data_minority_upsampled = resample(data_minority, replace=True, n_samples=n_samples, random_state=5)

    # Combine majority class with upsampled minority class
    data_upsampled = pd.concat([data_majority, data_minority_upsampled])

    data_upsampled.sort_index(inplace=True)

    # Display new class counts
    data_upsampled.target.value_counts()

    return data_upsampled

def normalize(x):

    scaler = StandardScaler()
    x_norm = scaler.fit_transform(x.values)
    x_norm = pd.DataFrame(x_norm, index=x.index, columns=x.columns)

    return x_norm


def scores(models, X, y):

    for model in models:
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc = roc_auc_score(y, y_pred)
        print("Accuracy Score: {0:0.2f} %".format(acc * 100))
        print("F1 Score: {0:0.4f}".format(f1))
        print("Area Under ROC Curve Score: {0:0.4f}".format(auc))

symbolin = 'NDX'  # SPX, DJI, NDX
data = get_data(symbolin)
data.tail(10)
data.describe()
ax = data['4. close'].plot(figsize=(9, 5))
ax.set_ylabel("Price ($)")
ax.set_xlabel("Time")
plt.show()
data_train = data['2012-01-01':'2018-01-01']
data_train = rebalance(data_train)
y = data_train.target
X = data_train.drop('target', axis=1)
X = normalize(X)
data_val = data['2018-01-01':]
y_val = data_val.target
X_val = data_val.drop('target', axis=1)
X_val = normalize(X_val)

# API DATA EXTRACTION AND PREPARATION

### API Data

X

X.to_csv('X.csv')

### Adding Change features in API data

X=pd.read_csv('X.csv')

X =pd.merge(X, change_feature, on='date')

X

y.head()
y.to_csv('y.csv')

y=pd.read_csv('y.csv')

y=pd.read_csv('y.csv')
y.columns = ['date', 'y']
y.head()

X =pd.merge(X, y, on='date')

X

### Dataset analysis

X [X .isnull().any(axis=1)].head()

import numpy as np
np.sum(X
       .isnull().any(axis=1))

X .isnull().any(axis=0)

X .info()

class_y=X.y
labels_X=X.drop(columns=['y','date'])

# MODEL BUILDING AND ANALYSIS

## SUPPORT VECTOR MACHINE MODEL

### SVM Support Vector Machine

from sklearn.svm import LinearSVC
y=y
X_train, X_test, y_train, y_test = train_test_split(labels_X, class_y, test_size=0.10, shuffle=True, random_state=2000)

svc=SVC(gamma="auto",random_state=5)
svc= svc.fit(X_train , y_train)
svc

y_pred1 = svc.predict(X_test)
print('Accuracy score= {:.2f}'.format(svc.score(X_test, y_test)*100))

### Precision, Recall, F1

from sklearn.metrics import classification_report, confusion_matrix


print('\n')
print("Precision, Recall, F1")
print('\n')
CR=classification_report(y_test, y_pred1)
print(CR)
print('\n')

### ROC CURVE

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_pred1)

roc_auc = auc(fpr, tpr)


plt.figure()

plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC CURVE')

plt.legend(loc="lower right")

plt.show()

## RANDOM FOREST MODEL

### Random Forest


Ran_For= RandomForestClassifier(n_estimators=1000,max_depth=30, random_state=100,max_leaf_nodes=1000)
Ran_For= Ran_For.fit(X_train , y_train)
Ran_For

y_pred2 = Ran_For.predict(X_test)
print('Accuracy score= {:.2f}'.format(Ran_For.score(X_test, y_test)*100))

### Precision, Recall, F1

from sklearn.metrics import classification_report, confusion_matrix


print('\n')
print("Precision, Recall, F1")
print('\n')
CR=classification_report(y_test, y_pred2)
print(CR)
print('\n')

### ROC CURVE

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_pred2)

roc_auc = auc(fpr, tpr)


plt.figure()

plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC CURVE')

plt.legend(loc="lower right")

plt.show()

## COMPARISION OF SVM AND RANDOM FOREST MODELS

### Comparison of Results

from prettytable import PrettyTable
x = PrettyTable()
print('\n')
print("Deatiled Performance of the all models")
x.field_names = ["Model", "Accuracy"]

x.add_row(["SVM", 0.74])
x.add_row(["RandomForestClassifier",0.84])
print(x)
print('\n')

x = PrettyTable()
print('\n')
print("Best Model.")
x.field_names = ["Model", "Accuracy"]
x.add_row(["RandomForestClassifier",0.84])
print(x)
print('\n')

### Best Accuracy of Random Forest Algorithm is 84%

# RECURSSIVE FEATURE SELECTION

## IMPORTING DEPENDENCIES

import pandas as pd
from sklearn.metrics import accuracy_score as AS
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score as RAS
from sklearn.metrics import f1_score as f1
from sklearn.model_selection import train_test_split as TTS
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
# %matplotlib inline
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

## RANDOM FOREST MODEL

### DATA PRE-PROCESSING

X_train, X_test, y_train, y_test = TTS(labels_X, class_y, test_size=0.10, shuffle=True, random_state=2000)

### EXTRACTING BEST FEATURES FROM DATASET

rfe=RFE(estimator=RFC(n_estimators=1000,max_depth=30, random_state=100,max_leaf_nodes=1000),step=2)

rfe=rfe.fit(X_train,y_train)

sel_features=pd.DataFrame({"Feature Name":list(X_train.columns),"Rank of feature":rfe.ranking_})
sel_features

rank1Features=list(sel_features[sel_features["Rank of feature"]==1]["Feature Name"])
rank1Features

### RANDOM FOREST MODEL TESTING AGAINST BEST FEATURES SELECTED

X=labels_X[rank1Features]
X_train, X_test, y_train, y_test = TTS(X,class_y, test_size=0.10, shuffle=True, random_state=2000)

Model=RFC(n_estimators=1000,max_depth=30, random_state=100,max_leaf_nodes=1000)

Model.fit(X_train,y_train)
prediction=Model.predict(X_test)

print("ACCURACY is : {:.2f}".format(AS(y_test,prediction)*100))

### Precision, Recall, F1

print('\n')
print("Precision, Recall, F1")
print('\n')
CR=classification_report(y_test, prediction)
print(CR)
print('\n')

### ROC CURVE

fpr, tpr, thresholds = roc_curve(y_test, prediction)

roc_auc = auc(fpr, tpr)


plt.figure()

plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC CURVE')

plt.legend(loc="lower right")

plt.show()

## SVM MODEL

### DATA PRE-PROCESSING

X_train, X_test, y_train, y_test = TTS(labels_X,class_y, test_size=0.10, shuffle=True, random_state=2000)

### EXTRACTING BEST FEATURES FROM DATASET

rfe=RFE(estimator=SVC(kernel="linear",gamma="auto",random_state=5),step=5)

rfe=rfe.fit(X_train,y_train)

sel_features=pd.DataFrame({"Feature Name":list(X_train.columns),"Rank of feature":rfe.ranking_})
sel_features

rank1Features=list(sel_features[sel_features["Rank of feature"]<10]["Feature Name"])
rank1Features

### SVM MODEL TESTING AGAINST BEST FEATURES SELECTED

X=labels_X[rank1Features]
X_train, X_test, y_train, y_test = TTS(X, class_y, test_size=0.10, shuffle=True, random_state=2000)

Model=SVC(gamma="auto",random_state=5)

Model.fit(X_train,y_train)
prediction=Model.predict(X_test)

print("ACCURACY is : {:.2f}".format(AS(y_test,prediction)*100))

### Precision, Recall, F1

print('\n')
print("Precision, Recall, F1")
print('\n')
CR=classification_report(y_test, prediction)
print(CR)
print('\n')

### ROC CURVE

fpr, tpr, thresholds = roc_curve(y_test, prediction)

roc_auc = auc(fpr, tpr)


plt.figure()

plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC CURVE')

plt.legend(loc="lower right")

plt.show()

## COMPARISION OF SVM AND RANDOM FOREST MODELS

### Comparison of Results


from prettytable import PrettyTable
x = PrettyTable()
print('\n')
print("Deatiled Performance of the all models")
x.field_names = ["Model", "Accuracy"]

x.add_row(["SVM", 74.24])
x.add_row(["RandomForestClassifier",85.59])
print(x)
print('\n')

x = PrettyTable()
print('\n')
print("Best Model.")
x.field_names = ["Model", "Accuracy"]
x.add_row(["RandomForestClassifier",85.59])
print(x)
print('\n')

## Accuracy of Random Forest Algorithm after FEATURE SELECTION is 85.59%

# TRAINING OF ALL DATA

### Now training all data (training&testing) on Random Forest Algorithm


class_y=X.y
labels_X=X.drop(columns=['y','date'])

### Random Forest

Ran_For= RandomForestClassifier(n_estimators=1000,max_depth=30, random_state=100,max_leaf_nodes=1000)
Ran_For= Ran_For.fit(labels_X , class_y)
Ran_For

### By Entring the Labels of target 1

data_for_prediction=pd.read_csv('prediction.csv')

data_for_prediction

### Prediction by using trained model

aa=Ran_For.predict(data_for_prediction)
if aa==1:
    print("stock prediction is 1")
else:
    print("stock prediction is 0")