import pandas as pd

df = pd.read_csv('data/bank-full.csv', sep=';')

print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(df['y'].value_counts())
print(df['y'].value_counts(normalize=True))
print(df.head())

job_subscription_rate = df.groupby('job')['y'].apply(lambda x: (x == 'yes').mean())
print("\nSubscription rate by job:")
print(job_subscription_rate.sort_values(ascending=False))