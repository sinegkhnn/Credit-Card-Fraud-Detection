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

## VERİNİN OKUNMASI :

```bash
data=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
```

```bash
data.head()
```

## BOŞ DEĞERLER:
```bash
data.isnull().sum()
```



