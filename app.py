import pandas as pd
import quandl
import math

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]
# remove redundant features

df['HL_%'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['%_CHANGE'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

# now having this data, let's define new `data frame`
df = df[['Adj. Close','HL_%', '%_CHANGE', 'Adj. Volume',]]

print(df.head())

forecast_col = 'Adj. Close'
# Replace NaNs instead of getting rid of data
df.fillna(-99999, inplace=True)
# predict 10% of our data frame
forecast_out = int(math.ceil(0.01*len(df)))

# LABELS

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.head())
