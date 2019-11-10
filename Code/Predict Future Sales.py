# Basic packages
# settings
import warnings

# Viz
import matplotlib.pyplot as plt  # basic plotting
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns  # for prettier plots
import statsmodels.api as sm
# TIME SERIES
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

sales = pd.read_csv("sales_train.csv")
item_cat = pd.read_csv("item_categories.csv")
item = pd.read_csv("items.csv")
sub = pd.read_csv("sample_submission.csv")
shops = pd.read_csv("shops.csv")
test = pd.read_csv("test.csv")
# sales.head(6)
# item_cat.head(6)
# item.head(6)
# sub.head(6)
# shops.head(6)
# test.head(6)

# formatting the date column correctly
sales.date = pd.to_datetime(sales.date, format='%d.%m.%Y')

monthly_sales = sales.groupby(["date_block_num", "shop_id", "item_id"])["date", "item_price", "item_cnt_day"].agg(
    {"date": ["min", 'max'], "item_price": "mean", "item_cnt_day": "sum"})

# number of items per cat
x = item.groupby(['item_category_id']).count()
x = x.sort_values(by='item_id', ascending=False)
x = x.iloc[0:10].reset_index()
x

# plot
plt.figure(figsize=(8, 4))
ax = sns.barplot(x.item_category_id, x.item_id, alpha=0.8)
plt.title("Items per Category")
plt.ylabel('# of items', fontsize=12)
plt.xlabel('Category', fontsize=12)
plt.show()

ts = sales.groupby(["date_block_num"])["item_cnt_day"].sum()
ts.astype('float')
plt.figure(figsize=(16, 8))
plt.title('Total Sales of the company')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(ts)

plt.figure(figsize=(16, 6))
plt.plot(ts.rolling(window=12, center=False).mean(), label='Rolling Mean');
plt.plot(ts.rolling(window=12, center=False).std(), label='Rolling sd');
plt.legend();

# multiplicative
res = sm.tsa.seasonal_decompose(ts.values, freq=12, model="multiplicative")
plt.figure(figsize=(30, 25))
fig = res.plot()
fig.show()


# Stationarity tests
def test_stationarity(timeseries):
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


test_stationarity(ts)

# to remove trend
from pandas import Series as Series


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# invert differenced forecast
def inverse_difference(last_ob, value):
    return value + last_ob


# ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()
# ts.astype('float')
plt.figure(figsize=(16, 16))
plt.subplot(311)
plt.title('Original')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(ts)
plt.subplot(312)
plt.title('After De-trend')
plt.xlabel('Time')
plt.ylabel('Sales')
new_ts = difference(ts)
plt.plot(new_ts)
plt.plot()

plt.subplot(313)
plt.title('After De-seasonalization')
plt.xlabel('Time')
plt.ylabel('Sales')
new_ts = difference(ts, 12)  # assuming the seasonality is 12 months long
plt.plot(new_ts)
plt.plot()

# now testing the stationarity again after de-seasonality
test_stationarity(new_ts)
