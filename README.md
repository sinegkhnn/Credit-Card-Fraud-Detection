Hazırlayan : Sine Gökhan

Proje : Kredi Kartı Dolandırıcılık Tespiti

Email : sineegkhnn@gmail.com
## KÜTÜPHANELERİN YÜKLENMESİ:

```bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
import warnings
warnings.filterwarnings('ignore')
```

## VERİNİN OKUNMASI (Kaggle'dan aldığımız datasetler ile) :

```bash
data=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
```

```bash
data.head()
```

## Verisetindeki veri düzgün olmadığı için boş verileri bulmalıyız:
```bash
data.isnull().sum()
```

Time      0
V1        0
V2        0
V3        0
V4        0
V5        0
V6        0
V7        0
V8        0
V9        0
V10       0
V11       0
V12       0
V13       0
V14       0
V15       0
V16       0
V17       0
V18       0
V19       0
V20       0
V21       0
V22       0
V23       0
V24       0
V25       0
V26       0
V27       0
V28       0
Amount    0
Class     0
dtype: int64

BU VERİ SETİNDE HİÇ BOŞ DEĞER OLMADIĞINI GÖRÜYORUZ.

```bash
Total_transactions = len(data)
normal = len(data[data.Class == 0])
fraudulent = len(data[data.Class == 1])
fraud_percentage = round(fraudulent/normal*100, 2)
print(cl('Total number of Trnsactions are {}'.format(Total_transactions), attrs = ['bold']))
print(cl('Number of Normal Transactions are {}'.format(normal), attrs = ['bold']))
print(cl('Number of fraudulent Transactions are {}'.format(fraudulent), attrs = ['bold']))
print(cl('Percentage of fraud Transactions is {}'.format(fraud_percentage), attrs = ['bold']))
```

Bu verilerle ilgili fark edeceğiniz en önemli şey, veri kümesinin bir özelliğe göre dengesiz olmasıdır. Bu nedenle, veri kümelerimize ait işlemlerin çoğunluğunun normal olduğunu ve işlemlerin yalnızca birkaç yüzdesinin hileli olduğunu görebiliyoruz.

<img src="[./src/images/github-profile-readme-generator.gif](https://camo.githubusercontent.com/055be2eea5a7ab2dbad79044b4fb9d6f0e201f51efda519da0fe0b0254c5d451/68747470733a2f2f6d69726f2e6d656469756d2e636f6d2f6d61782f3532342f312a52695a6b543235444e45386f312d4d6f3367744175772e706e67)" />

VERİ HAKKINDA BİLGİLERE GÖZ ATALIM

```bash
data.info()
```

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 284807 entries, 0 to 284806
Data columns (total 31 columns):
Time      284807 non-null float64
V1        284807 non-null float64
V2        284807 non-null float64
V3        284807 non-null float64
V4        284807 non-null float64
V5        284807 non-null float64
V6        284807 non-null float64
V7        284807 non-null float64
V8        284807 non-null float64
V9        284807 non-null float64
V10       284807 non-null float64
V11       284807 non-null float64
V12       284807 non-null float64
V13       284807 non-null float64
V14       284807 non-null float64
V15       284807 non-null float64
V16       284807 non-null float64
V17       284807 non-null float64
V18       284807 non-null float64
V19       284807 non-null float64
V20       284807 non-null float64
V21       284807 non-null float64
V22       284807 non-null float64
V23       284807 non-null float64
V24       284807 non-null float64
V25       284807 non-null float64
V26       284807 non-null float64
V27       284807 non-null float64
V28       284807 non-null float64
Amount    284807 non-null float64
Class     284807 non-null int64
dtypes: float64(30), int64(1)
memory usage: 67.4 MB

```bash
data.describe().T.head()
```

```bash
data.shape
```
(284807, 31)

GÖRÜYORUZ Kİ 284807 SATIR VE 31 SÜTUN VAR.

```bash
data.columns
```

Index(['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
       'Class'],
      dtype='object')
## DOLANDIRICILIK VERİLERİ VE GERÇEK VERİLER:


```bash
fraud_cases=len(data[data['Class']==1])
```

```bash
print('Dolandırıcılık Verileri Sayısı:',fraud_cases)
```

```bash
non_fraud_cases=len(data[data['Class']==0])
```

```bash
print('Dolandırıcılık Verileri Sayısı:',non_fraud_cases)
```

```bash
fraud=data[data['Class']==1]
```

```bash
genuine=data[data['Class']==0]

```

```bash
fraud.Amount.describe()
```
count     492.000000
mean      122.211321
std       256.683288
min         0.000000
25%         1.000000
50%         9.250000
75%       105.890000
max      2125.870000
Name: Amount, dtype: float64

```bash
genuine.Amount.describe()
```

count    284315.000000
mean         88.291022
std         250.105092
min           0.000000
25%           5.650000
50%          22.000000
75%          77.050000
max       25691.160000
Name: Amount, dtype: float64

```bash
rcParams['figure.figsize'] = 16, 8
f,(ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(genuine.Time, genuine.Amount)
ax2.set_title('Genuine')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()

```

## MODELLERİMİZE BAKALIM:
```bash
from sklearn.model_selection import train_test_split
```

## Model 1:

```bash
X=data.drop(['Class'],axis=1)
```
```bash
y=data['Class']
```
```bash
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=123)
```
```bash
from sklearn.ensemble import RandomForestClassifier
```
```bash
rfc=RandomForestClassifier()
```
```bash
model=rfc.fit(X_train,y_train)
```
```bash
prediction=model.predict(X_test)
```
```bash
from sklearn.metrics import accuracy_score
```
```bash
accuracy_score(y_test,prediction)
```
0.9995786664794073

## Model 2:

```bash
from sklearn.linear_model import LogisticRegression
```
```bash
X1=data.drop(['Class'],axis=1)
```
```bash
y1=data['Class']
```
```bash
X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.3,random_state=123)
```
```bash
lr=LogisticRegression()
```
```bash
model2=lr.fit(X1_train,y1_train)
```
```bash
prediction2=model2.predict(X1_test)
```
```bash
accuracy_score(y1_test,prediction2)
```
0.9988764439450862

## Model 3:

```bash
xgb = XGBClassifier(max_depth = 4)
xgb.fit(X_train, y_train)
xgb_yhat = xgb.predict(X_test)
```

```bash
print('XGBoost modelinin doğruluk puanı {}'.format(accuracy_score(y_test, xgb_yhat)))
```
XGBoost modelinin doğruluk puanı 0.9995211561901445

```bash
print('XGBoost modelinin F1 puanı {}'.format(f1_score(y_test, xgb_yhat)))
```
XGBoost modelinin F1 puanı 0.8421052631578947

```bash
from sklearn.tree import DecisionTreeRegressor
```
```bash
X2=data.drop(['Class'],axis=1)
```
```bash
y2=data['Class']
```
```bash
dt=DecisionTreeRegressor()
```
```bash
X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.3,random_state=123)
```
```bash
model3=dt.fit(X2_train,y2_train)
```
```bash
prediction3=model3.predict(X2_test)
```
```bash
accuracy_score(y2_test,prediction3)
```
0.999133925541004

## SONUÇ : 
Görüyoruz ki kredi kartı dolandırıcılık tespitinde modellerimize göre %99,95 doğruluk elde ettik.
