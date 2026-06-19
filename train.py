import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Load and preprocess
df = pd.read_csv('data/bank-full.csv', sep=';')
df['y'] = df['y'].map({'yes': 1, 'no': 0})

categorical_cols = ['job', 'marital', 'education', 'default',
                     'housing', 'loan', 'contact', 'month', 'poutcome']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df_encoded.drop('y', axis=1).astype(int)
y = df_encoded['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale for Logistic Regression only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train all three models
log_reg = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
log_reg.fit(X_train_scaled, y_train)
print("Logistic Regression trained.")

rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
print("Random Forest trained.")

xgb = XGBClassifier(scale_pos_weight=7.3, random_state=42, eval_metric='logloss')
xgb.fit(X_train, y_train)
print("XGBoost trained.")

# Evaluate all three
log_reg_preds = log_reg.predict(X_test_scaled)
rf_preds = rf.predict(X_test)
xgb_preds = xgb.predict(X_test)

print("\n=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, log_reg_preds))
print("F1 Score:", f1_score(y_test, log_reg_preds))
print(classification_report(y_test, log_reg_preds))

print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, rf_preds))
print("F1 Score:", f1_score(y_test, rf_preds))
print(classification_report(y_test, rf_preds))

print("\n=== XGBoost ===")
print("Accuracy:", accuracy_score(y_test, xgb_preds))
print("F1 Score:", f1_score(y_test, xgb_preds))
print(classification_report(y_test, xgb_preds))


import numpy as np

# Feature importance from XGBoost
feature_names = X.columns.tolist()
importances = xgb.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n=== Top 10 Feature Importances (XGBoost) ===")
for i in range(10):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

    # Get XGBoost predictions and probabilities on test set
xgb_probs = xgb.predict_proba(X_test)[:, 1]  # probability of class 1 (yes)
xgb_preds_all = xgb.predict(X_test)

# Convert X_test back to dataframe with column names for readability
X_test_df = pd.DataFrame(X_test, columns=feature_names)
X_test_df['predicted'] = xgb_preds_all
X_test_df['probability'] = xgb_probs
X_test_df['actual'] = y_test.values

# Pick samples: at least 2 predicted yes, 2 predicted no
predicted_yes = X_test_df[X_test_df['predicted'] == 1].head(3)
predicted_no = X_test_df[X_test_df['predicted'] == 0].head(2)
samples = pd.concat([predicted_yes, predicted_no])

# Print readable summary
print("\n=== 5 Sample Customer Predictions ===")
for i, (_, row) in enumerate(samples.iterrows()):
    print(f"\n--- Customer {i+1} ---")
    print(f"Age: {row['age']}")
    print(f"Balance: {row['balance']}")
    print(f"Duration: {row['duration']}")
    print(f"Campaign contacts: {row['campaign']}")
    print(f"Has housing loan: {'Yes' if row['housing_yes'] == 1 else 'No'}")
    print(f"Previous outcome success: {'Yes' if row['poutcome_success'] == 1 else 'No'}")
    print(f"Prediction: {'SUBSCRIBE' if row['predicted'] == 1 else 'NOT SUBSCRIBE'}")
    print(f"Probability of subscribing: {row['probability']:.2%}")
    print(f"Actual outcome: {'SUBSCRIBE' if row['actual'] == 1 else 'NOT SUBSCRIBE'}")



# Save model, scaler, and feature names
import joblib
import json

joblib.dump(xgb, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModel and scaler saved.")

feature_names = X.columns.tolist()
with open('feature_names.json', 'w') as f:
    json.dump(feature_names, f)
print("Feature names saved.")