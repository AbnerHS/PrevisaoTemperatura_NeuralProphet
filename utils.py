import pandas as pd

def to_numeric(x):
  x = str(x)
  x = x.replace(',','.')
  return float(x)

def preprocess(df):
  df.columns = ["Data","Temperatura"]
  df.Data = pd.to_datetime(df.Data, dayfirst=True)
  df.dtypes

  df.index = pd.to_datetime(df.Data, format='%m-%d-%Y %H:i')
  df['Temperatura'] = pd.to_numeric(df['Temperatura'].apply(to_numeric))
  df.drop("Data",axis=1,inplace=True)
  return df

def preencherZero(df):
  num_cols = len(list(df.columns.values))
  for col in range(num_cols):
    df.iloc[:,col] = df.iloc[:,col].fillna(df.iloc[:,col].mean())
  return df