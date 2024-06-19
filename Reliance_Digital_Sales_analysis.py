import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

filepath = "wetransfer_campaign_data-csv_2024-04-13_0433"
df_transaction = pd.read_csv(f"{filepath}/Customer_Transaction_Data.csv")
df_customer_data = pd.read_csv(f"{filepath}/Customer_Master_Data.csv")
df_campaign = pd.read_csv(f"{filepath}/Campaign_Data.csv")
df_reviews = pd.read_excel(f"{filepath}/CustFeedback.xlsx")
df_delivery = pd.read_csv(f"{filepath}/Customer_Delivery_Data.csv")

## Performing EDA on Transaction Data

transaction_sample = pd.concat([df_transaction.head(100), df_transaction.tail(100)])

# Identifying the column names and their datatypes
df_transaction.info()

# Checking % of NULL values in a particular column
"""
1. If NULL percentage < 3%, use dropna()
2. If NULL percentage > 3% and <= 40%, use fill-na
3. If NULL percentage > 40%, do segregation of NULL and non-NULL data
"""
(df_transaction['StoreCity'].isnull().sum()) / len(df_transaction['StoreCity'])*100
# Removing all NaN values from df, except for the specified columns
df_trans_new = df_transaction.dropna(subset=df_transaction.columns.difference(["ReturnFlag","StoreCity","StoreState","StorePincode"]))
df_trans_new.info()

transaction_sample = pd.concat([df_trans_new.head(100), df_trans_new.tail(100)])

# Detecting Outliers:
"""
Below are the 3 approaches, which gives the same result. (I will use IQR)
1. Inter Quartile Range (IQR)
	q1 = df["column"].quantile(0.25)
	q3 = df["column"].quantile(0.75)
    IQR = q3 - q1

	lower_limit_outlier = q1-1.5*IQR
	upper_limit_outlier = q3+1.5*IQR

	outlier_data = df[(df['column']<=lower_limit_outlier) | (df['column']>=upper_limit_outlier)]

2. Standard Deviation (Distance of a datapoint from mean/avg)

	mean_height = df2['height'].mean() #mean of height
	std_height = df2['height'].std()  #standard deviation.

	lower_limit_height = mean_height - 3 * std_height
	upper_limit_height = mean_height + 3 * std_height

	outlier_data_new = df2[(df2['height'] < lower_limit_height) | (df2['height'] > upper_limit_height)]

3. Z score
	(x-mean)/std -> where x is the datapoint
	df2['Z-Score-column'] = (df2['column']-df2['column'].mean()) / df2['column'].std()

	-The Z-Score if > 3 will be considered as an upper limit outlier
	-The Z-Score if < -3 will be considered as an lower limit outlier
	df2[(df2['Z-Score-column'] <-3) | (df2['Z-Score-column'] > 3)]
"""
q1 = df_trans_new["SaleValue"].quantile(0.25)
q3 = df_trans_new["SaleValue"].quantile(0.75)
IQR = q3 - q1

lower_limit_outlier = q1-1.5*IQR
upper_limit_outlier = q3+1.5*IQR

outlier_data = df_trans_new[(df_trans_new['SaleValue']<=lower_limit_outlier) | (df_trans_new['SaleValue']>=upper_limit_outlier)]



# Skewness (Used to check biasness of the data. Data must be normally spread.)

sns.displot(df_trans_new['SaleValue'],kde=True)
sns.kdeplot(df_trans_new['SaleValue'])

df_trans_new['SaleValue'].skew() #Here, we get o/p as 2.144569688992913, which indicates the data is HIGHLY POSITIVELY SKEWED.

# Convertng object to datetime
df_trans_new['OrderDate'] = pd.to_datetime(df_trans_new['OrderDate'])
df_trans_new['Month'] = df_trans_new['OrderDate'].dt.month
df_trans_new['Year'] = df_trans_new['OrderDate'].dt.year

df_trans_new["MerchClassDescription"].value_counts()

#1
counts = df_trans_new.groupby(['OrderDate', 'MerchClassDescription']).size().unstack(fill_value=0)

#2
# Define the Diwali period for each year
diwali_dates = {
    2019: ('2019-11-14', '2019-11-15'),
    2020: ('2020-11-04', '2020-11-05'),
    2021: ('2021-10-24', '2021-10-26')
}

# Calculate sales revenue during Diwali for each year and each Merchant Class
diwali_sales_percentage = {}
for year, (start_date, end_date) in diwali_dates.items():
    diwali_sales = df_trans_new[(df_trans_new['OrderDate'] >= start_date) & (df_trans_new['OrderDate'] <= end_date)]
    total_diwali_sales = diwali_sales.groupby('MerchClassDescription')['SaleValue'].sum()

    annual_sales = df_trans_new[df_trans_new['OrderDate'].dt.year == year]
    total_annual_sales = annual_sales.groupby('MerchClassDescription')['SaleValue'].sum()

    # Calculate percentage of Diwali sales compared to total annual sales
    diwali_sales_percentage[year] = (total_diwali_sales / total_annual_sales) * 100

# Convert the dictionary to DataFrame
diwali_sales_percentage_df = pd.DataFrame(diwali_sales_percentage)

## Product Segmentation

merchant_categories = ["TV LCD", "TV Services", "Gaming Laptops", "Smart Phones (OS Based)", "Tablets & Detachables"]
filtered_df = df_trans_new[df_trans_new['MerchClassDescription'].isin(merchant_categories)]

sample_filtered = filtered_df[:1000]

# Convert OrderDate to datetime format and extract the year
filtered_df['OrderDate'] = pd.to_datetime(filtered_df['OrderDate'])
filtered_df['Year'] = filtered_df['OrderDate'].dt.year

# Calculate price percentile for each product
filtered_df['ProductType'] = filtered_df['SaleValue'].transform(lambda x: pd.qcut(x, q=[0, 0.33, 0.66, 1], labels=['Value', 'Mainstream', 'Premium']))
sample_filtered = filtered_df[:1000]

# Group by year, product description, and calculate total sales volume and revenue
summary = filtered_df.groupby(['Year', 'ProductType']).agg(
    SalesVolume=('OrderedQuantity', 'sum'),
    SalesRevenue=('SaleValue', 'sum')
).reset_index()



## Product Assortment

brick_mortar_df = filtered_df[filtered_df['Ecom_BnM_Indicator'] == 'B&M']

# Filter data for the most recent 6 months
recent_data = brick_mortar_df[brick_mortar_df['OrderDate'] >= brick_mortar_df['OrderDate'].max() - pd.DateOffset(months=6)]

# Group by store, year, month, merchant category, and price percentile
grouped_df = recent_data.groupby(['StoreCode', 'Year', 'Month', 'MerchCategoryDescription', 'ProductType']).agg({'OrderedQuantity': 'sum', 'SaleValue': 'sum'})
gs = grouped_df.head(50).reset_index()

# Analyze the distribution of sales for each store
top_5_stores = recent_data.groupby('StoreCode').sum()['SaleValue'].nlargest(5).index
top_5_store_data = recent_data[recent_data.index.get_level_values('StoreCode').isin(top_5_stores)]
t5 = top_5_store_data.tail().reset_index()

###########################################################################



df = df_campaign.merge(df_customer_data, on='CustID').merge(df_trans_new, on='CustID')

df['Campaign_Exec_Date'] = pd.to_datetime(df['Campaign_Exec_Date'])
df['OrderDate'] = pd.to_datetime(df['OrderDate'])

# Create the Outcome Variable
df.columns
df['Outcome'] = df.apply(lambda x: 1 if (x['OrderDate'] <= x['Campaign_Exec_Date'] + pd.DateOffset(months=1)) > 0 else 0, axis=1)
df_sample = pd.concat([df.head(100), df.tail(100)])

df['Outcome'].value_counts()
