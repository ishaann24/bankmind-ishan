import pandas as pd

# Load the data
df = pd.read_csv('data/bank-full.csv', sep=';')

# Step 1: Encode target column
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# Step 2: One-hot encode the categorical columns
categorical_cols = ['job', 'marital', 'education', 'default',
                     'housing', 'loan', 'contact', 'month', 'poutcome']

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Step 3: Separate features (X) from target (y)
X = df_encoded.drop('y', axis=1)
X = X.astype(int)
y = df_encoded['y']

# Sanity checks - print these so we can see what happened
print("Original shape:", df.shape)
print("Encoded shape:", df_encoded.shape)
print("X shape:", X.shape)
print("y shape:", y.shape)
print("\nFirst 5 rows of X:")
print(X.head())
print("\nColumn names in X:")
print(X.columns.tolist())


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Train y distribution:\n", y_train.value_counts(normalize=True))
print("Test y distribution:\n", y_test.value_counts(normalize=True))