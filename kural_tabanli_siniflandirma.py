import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv("/Users/eminebozkurt/Desktop/vbo/Week2/hw3/Kural_Tabanli_Siniflandirma/Kural_Tabanli_Siniflandirma/persona.csv")
df.head()
df.info()

# PRICE – Müşterinin harcama tutarı
# SOURCE – Müşterinin bağlandığı cihaz türü
# SEX – Müşterinin cinsiyeti
# COUNTRY – Müşterinin ülkesi AGE – Müşterinin yaşı

################################
# GOREV 1
################################


# Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
df = pd.read_csv("/Users/eminebozkurt/Desktop/vbo/Week2/hw3/Kural_Tabanli_Siniflandirma/Kural_Tabanli_Siniflandirma/persona.csv")
df.head()


# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?
df["SOURCE"].nunique()
df["SOURCE"].value_counts()

# Soru 3: Kaç unique PRICE vardır?
df["PRICE"].nunique()

# Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
df["PRICE"].value_counts()

# Soru 5: Hangi ülkeden kaçar tane satış olmuş?
df["COUNTRY"].value_counts()
df["COUNTRY"].value_counts(ascending=False, normalize=True) # hoca

# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?
df.head()
df.groupby("COUNTRY").agg({"PRICE": "sum"})

# Soru 7: SOURCE türlerine göre satış sayıları nedir?
df["SOURCE"].value_counts()
df.groupby("SOURCE").agg({"PRICE": "count"}) # hoca
# Soru 8: Ülkelere göre PRICE ortalamaları nedir?
df.groupby("COUNTRY").agg({"PRICE": "mean"})

# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?
df.groupby("SOURCE").agg({"PRICE": "mean"})

# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
df.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE": "mean"})
df.pivot_table("PRICE", "COUNTRY", "SOURCE", aggfunc="mean")

# Görev 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}).head()
df.pivot_table("PRICE",  ["COUNTRY", "SOURCE", "SEX", "AGE"], aggfunc="mean").head()

# Görev 3: Çıktıyı PRICE’a göre sıralayınız.
# • Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE’a göre uygulayınız.
# • Çıktıyı agg_df olarak kaydediniz.
agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", axis=0, ascending=False)
agg_df.head()

# Görev 4: Indekste yer alan isimleri değişken ismine çeviriniz.
# • Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler index isimleridir.
# Bu isimleri değişken isimlerine çeviriniz.
agg_df.head()
agg_df.index
agg_df.reset_index(inplace=True)

# Görev 5: Age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz.
# • Age sayısal değişkenini kategorik değişkene çeviriniz.
# • Aralıkları ikna edici şekilde oluşturunuz.
# • Örneğin: ‘0_18', ‘19_23', '24_30', '31_40', '41_70'
agg_df.head()
agg_df["AGE"].dtypes # dtype('int64')
agg_df["AGE"] = agg_df["AGE"].astype("category")

bins = [df["AGE"].min() - 1, 18, 23, 30, 40, df["AGE"].max()]
type(bins)# list
labels = [str(df["AGE"].min()) + "_18", "19_23", "24_30", "31_40", "41_" + str(df["AGE"].max())]
type(labels)# list
agg_df['AGE_CAT'] = pd.cut(agg_df['AGE'], bins=bins, labels=labels)
agg_df.head()
df.head()


# Görev 6: Yeni seviye tabanlı müşterileri (persona) tanımlayınız.
# • Yeni seviye tabanlı müşterileri (persona) tanımlayınız ve veri setine değişken olarak ekleyiniz.
# • Yeni eklenecek değişkenin adı: customers_level_based
# • Önceki soruda elde edeceğiniz çıktıdaki gözlemleri bir araya getirerek customers_level_based değişkenini oluşturmanız gerekmektedir.

agg_df["CUSTOMERS_LEVEL_BASED"] = agg_df[["COUNTRY", "SOURCE", "SEX", "AGE_CAT"]].apply(lambda x: '_'.join(x.values).upper(), axis=1)
agg_df.head()
df_persona = agg_df.groupby("CUSTOMERS_LEVEL_BASED").agg({"PRICE": "mean"})
df_persona = df_persona.sort_values("PRICE", ascending=False)
df_persona.head()


# Görev 7: Yeni müşterileri (personaları) segmentlere ayırınız.
# • Yeni müşterileri (Örnek: USA_ANDROID_MALE_0_18) PRICE’a göre 4 segmente ayırınız.
# • Segmentleri SEGMENT isimlendirmesi ile değişken olarak agg_df’e ekleyiniz.
# • Segmentleri betimleyiniz (Segmentlere göre group by yapıp price mean, max, sum’larını alınız).

df_persona["SEGMENT"] = pd.qcut(df_persona['PRICE'], 4, ["D", "C", "B", "A"])
df_persona.head()
df_persona.reset_index(inplace=True)
df_persona.groupby("SEGMENT").agg({"PRICE": ["mean", "max", "sum"]})
# df_segment = df_persona.groupby('SEGMENT').mean("PRICE").reset_index().sort_values("SEGMENT", ascending=False)



# Görev 8: Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini tahmin ediniz.
# • 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
# • 35 yaşında IOS kullanan bir Fransız kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
# ipucu
new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["CUSTOMERS_LEVEL_BASED"] == new_user]





def AGE_CAT(age):
    if age <= 18:
        AGE_CAT = "0_18"
        return AGE_CAT
    elif (age > 18 and age <= 23):
        AGE_CAT = "19_23"
        return AGE_CAT
    elif (age > 23 and age <= 30):
        AGE_CAT = "24_30"
        return AGE_CAT
    elif (age > 30 and age <= 40):
        AGE_CAT = "31_40"
        return AGE_CAT
    elif (age > 40 and age <= 70):
        AGE_CAT = "41_70"
        return AGE_CAT


AGE_CAT(30)
df_persona.head()

# df["AGE"].min() - 1, 18, 23, 30, 40, df["AGE"].max()
# labels = [str(df["AGE"].min()) + "18", "19_23", "24_30", "31_40", "41_" + str(df["AGE"].max())]


def new_user_classification():
    COUNTRY = input("Country name (USA/EUR/BRA/DEU/TUR/FRA): ")
    SOURCE = input("Phone (IOS/ANROID): ")
    SEX = input("Sex (FEMALE/MALE): ")
    AGE = int(input("Age: "))
    AGE_SEGMENT = AGE_CAT(AGE)
    new_user = (COUNTRY + '_' + SOURCE + '_' + SEX + '_' + AGE_SEGMENT).upper()
    print(new_user)
    if df_persona[df_persona["CUSTOMERS_LEVEL_BASED"] == new_user].shape[0] > 0:
        print("Segment:" + df_persona[df_persona["CUSTOMERS_LEVEL_BASED"] == new_user].loc[:, "SEGMENT"].values[0])
        print("Price:" + str(df_persona[df_persona["CUSTOMERS_LEVEL_BASED"] == new_user].loc[:, "PRICE"].values[0]))
    else:
        print("Unknown User! Try Again!")


df_persona.head()


new_user_classification()
df_persona.head()

new_user = "EUR_IOS_MALE_36_45"
agg_df[agg_df["CUSTOMERS_LEVEL_BASED"] == new_user]
new_user = "FRA_ANDROID_MALE_15_18"


new_user = "TUR_ANDROID_FEMALE_24_35"
df_persona[df_persona["CUSTOMERS_LEVEL_BASED"] == "new_user"]["SEGMENT"].values[0]
df_persona[df_persona["CUSTOMERS_LEVEL_BASED"] == new_user]["PRICE"].values[0]


new_user = "TUR_ANDROID_FEMALE_24_35"
df_persona[df_persona["CUSTOMERS_LEVEL_BASED"] == new_user]["SEGMENT"].values[0]
df_persona[df_persona["CUSTOMERS_LEVEL_BASED"] == new_user]["PRICE"].values[0]

new_user = "FRA_ANDROID_FEMALE_24_35"
df_persona[df_persona["CUSTOMERS_LEVEL_BASED"] == new_user]["SEGMENT"].values[0]
df_persona[df_persona["CUSTOMERS_LEVEL_BASED"] == new_user]["PRICE"].values[0]
