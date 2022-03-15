!pip install yellowbrick

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder

#Part 1 
#Step 1
df_ = pd.read_csv("flo_data_20K.csv")
df = df_

df.head()

df.isnull().sum()

df.info()
#Tarih değişkenleri object türünde bunları düzeltmemiz gerekir.
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.info()
df.describe().T

#Aykırı değer problemi gözükmektedir. Bunları da düzeltelim.
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def grab_col_names(dataframe, cat_th=7, car_th=8):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


for col in num_cols:
    replace_with_thresholds(df, col)

#Step 2
# Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi
import datetime as dt
df["last_order_date"].max() # 2021-05-30
analysis_date = dt.datetime(2021,6,1)
df.head()

# customer_id, recency, frequnecy ve monetary değerlerinin yer aldığı yeni bir rfm dataframe
df["customer_id"] = df["master_id"]
df["recency"] = (analysis_date - df["last_order_date"]).astype('timedelta64[D]')/7
df["tenure"] = ((analysis_date - df["first_order_date"]).astype('timedelta64[D]'))/7


#Değişken olarak seçmek istediklerim, müşterilerin alışveriş hareketlerinde belirleyici olacağını düşündüğüm değişkenler olacaktır.
#Bu bağlamda; recency, tenure, order channel, order num total online & offline, customer value total online & offline.


df = df[["customer_id","order_channel","recency","tenure","order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_online","customer_value_total_ever_offline"]]

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if "customer_id" not in col]

#Part 2
#Step 1
sc = MinMaxScaler((0,1))
k_df = sc.fit_transform(df[num_cols])


#Step 2


kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2,20))
elbow.fit(k_df)
elbow.show()

elbow.elbow_value_

#Step 3
kmeans = KMeans(n_clusters = elbow.elbow_value_).fit(k_df)
kmeans.n_clusters

kmeans.cluster_centers_
kmeans.labels_
clusters = kmeans.labels_

df["cluster"] = clusters
df.head()
#Step 4
df.groupby("cluster").agg(["count","mean","median"])

#3 numaralı kümelemeye bakıldığında customer_value_total_ever_online değişkeni en fazla kar getiren grup olduğu gözlenmektedir.
#3 numaralı gruba yönelik kaynak aktarımları yapılabilir.
#Harcamaların totaline bakılırsa 1 ve 5 numaralı kümeler birleştirilebilir.
#Yine harcama detaylarına ve recency değişkeninden yola çıkarak bir yorum yaparsak recency değeri birbirine yakın olan
#3 ve 4 numaralı kümeler toplam harcamada en fazla olan grup olmayı sürdüreceklerdir.


#Part 3 
#Step 1
hc_average = linkage(k_df,"average")
plt.figure(figsize=(7,5))
plt.title("Dendogram")
dend = dendrogram(hc_average,truncate_mode="lastp",p=10,
                show_contracted=True,
                leaf_font_size=10)
plt.axhline(y=1.1,color = "r", linestyle="--")
plt.show()

#Küme sayısını 4 olarak belirliyoruz.

from sklearn.cluster import AgglomerativeClustering
h_cluster = AgglomerativeClustering(n_clusters=4,linkage="average")
h_clusters = h_cluster.fit_predict(k_df)

df["hi_cluster"] = h_clusters
df.groupby("hi_cluster").agg(["count","mean","median"])

#Step 3

#3 numaralı kümede tek eleman olduğu değer olduğu için segmentasyon açısından değer içermeyecektir.
#0 numaralı küme ise online alışveriş tutarında en değerli grup bu grup için online için kampanyalar düzenlenebilir.
#1 numaralı küme terk etmiş müşteriler olabilir. Ve bu müşteriler harcamalarının büyük kısmını mağazaya gelerek gerçekleştirmektedirler.
#2 numaralı küme müşterilerin genelinin bulunduğu gruptur. Alışveriş harcamalarının büyük kısmını online olarak gerçekleştirmektedirler. Online olarak kaynak planlanması yapılabilir.













