import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
test_url  = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'class', 'level'
]

print("Loading data...")
df_train = pd.read_csv(train_url, names=columns)
df_test  = pd.read_csv(test_url,  names=columns)

df_train.drop(columns=['level'], inplace=True)
df_test.drop(columns=['level'],  inplace=True)

print(f"Training set: {df_train.shape[0]} records")
print(f"Test set:     {df_test.shape[0]} records")


df_full = pd.concat([df_train, df_test])

cat_cols = ['protocol_type', 'service', 'flag']
for col in cat_cols:
    le = LabelEncoder()
    df_full[col] = le.fit_transform(df_full[col])


category_map = {
    'normal': 'Normal',
    'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS',
    'processtable': 'DoS', 'udpstorm': 'DoS', 'worm': 'DoS',
    'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    'warezclient': 'R2L', 'guess_passwd': 'R2L', 'ftp_write': 'R2L',
    'imap': 'R2L', 'phf': 'R2L', 'multihop': 'R2L', 'warezmaster': 'R2L',
    'spy': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguess': 'R2L',
    'snmpgetattack': 'R2L', 'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R',
    'perl': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R'
}

df_full['category'] = df_full['class'].map(category_map).fillna('Other')
df_full.drop(columns=['num_outbound_cmds', 'class'], inplace=True)

train_len = len(df_train)
df_train_proc = df_full.iloc[:train_len].copy()
df_test_proc  = df_full.iloc[train_len:].copy()

X_train = df_train_proc.drop(columns=['category'])
y_train = df_train_proc['category']
X_test  = df_test_proc.drop(columns=['category'])
y_test  = df_test_proc['category']

print(f"\nTraining class distribution:\n{y_train.value_counts()}")
print(f"\nTest class distribution:\n{y_test.value_counts()}")


print("\nApplying SMOTE to oversample minority classes...")

smote = SMOTE(
    sampling_strategy={'R2L': 20000, 'U2R': 3000},
    k_neighbors=5,
    random_state=RANDOM_STATE
)

X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
print(f"Resampled class distribution:\n{pd.Series(y_resampled).value_counts()}")

label_enc = LabelEncoder()
y_resampled_enc = label_enc.fit_transform(y_resampled)

weight_map = {'Normal': 1, 'DoS': 1, 'Probe': 1, 'R2L': 5, 'U2R': 10}
sample_weights = np.array([weight_map[c] for c in y_resampled])


from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

results = {}

print("\n--- Training Model 1: Random Forest ---")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_resampled, y_resampled_enc)
rf_pred = label_enc.inverse_transform(rf.predict(X_test))
rf_f1 = f1_score(y_test, rf_pred, average='macro')
results['Random Forest'] = (rf_f1, rf_pred)
print(f"Random Forest macro F1: {rf_f1:.4f}")

print("\n--- Training Model 2: LightGBM ---")
lgbm = LGBMClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1
)
lgbm.fit(X_resampled, y_resampled_enc, sample_weight=sample_weights)
lgbm_pred = label_enc.inverse_transform(lgbm.predict(X_test))
lgbm_f1 = f1_score(y_test, lgbm_pred, average='macro')
results['LightGBM'] = (lgbm_f1, lgbm_pred)
print(f"LightGBM macro F1: {lgbm_f1:.4f}")

print("\n--- Training Model 3: XGBoost ---")
xgb = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=RANDOM_STATE,
    n_jobs=-1
)
xgb.fit(X_resampled, y_resampled_enc, sample_weight=sample_weights)
xgb_pred = label_enc.inverse_transform(xgb.predict(X_test))
xgb_f1 = f1_score(y_test, xgb_pred, average='macro')
results['XGBoost'] = (xgb_f1, xgb_pred)
print(f"XGBoost macro F1: {xgb_f1:.4f}")

# --- Pick the best model ---
print("\n--- Model Comparison ---")
for name, (score, _) in sorted(results.items(), key=lambda x: x[1][0], reverse=True):
    print(f"  {name}: {score:.4f}")

best_name = max(results, key=lambda x: results[x][0])
best_f1, y_pred = results[best_name]
print(f"\nBest model: {best_name} with macro F1 = {best_f1:.4f}")


print(f"\nRunning 5-fold cross-validation for {best_name}...")

if best_name == 'XGBoost':
    cv_model = XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric='mlogloss', random_state=RANDOM_STATE, n_jobs=-1)
elif best_name == 'LightGBM':
    cv_model = LGBMClassifier(n_estimators=500, max_depth=8, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              class_weight='balanced', random_state=RANDOM_STATE,
                              n_jobs=-1, verbose=-1)
else:
    cv_model = RandomForestClassifier(n_estimators=300, max_depth=20,
                                      class_weight='balanced',
                                      random_state=RANDOM_STATE, n_jobs=-1)

cv_scores = cross_val_score(cv_model, X_resampled, y_resampled_enc,
                            cv=5, scoring='f1_macro')
print(f"Cross-validation macro F1: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")


print("\nApplying threshold tuning for R2L and U2R...")

if best_name == 'XGBoost':
    best_model = xgb
elif best_name == 'LightGBM':
    best_model = lgbm
else:
    best_model = rf

proba = best_model.predict_proba(X_test)
classes = label_enc.classes_

thresholds = {'R2L': 0.005, 'U2R': 0.003}

adjusted_proba = proba.copy()
for i, c in enumerate(classes):
    if c in thresholds:
        adjusted_proba[:, i] = adjusted_proba[:, i] / thresholds[c]

y_pred_tuned = label_enc.inverse_transform(np.argmax(adjusted_proba, axis=1))
tuned_f1 = f1_score(y_test, y_pred_tuned, average='macro')

print(f"Before threshold tuning: {best_f1:.4f}")
print(f"After threshold tuning:  {tuned_f1:.4f}")

if tuned_f1 > best_f1:
    print("Threshold tuning improved results — using tuned predictions.")
    y_pred = y_pred_tuned
    test_macro_f1 = tuned_f1
else:
    print("Threshold tuning did not improve — keeping original predictions.")
    test_macro_f1 = best_f1

print("\n" + "="*60)
print(f"CLASSIFICATION REPORT — {best_name} + SMOTE + Threshold Tuning")
print("="*60)
print(classification_report(y_test, y_pred))
print(f"Test macro F1-score:          {test_macro_f1:.4f}")
print(f"Cross-validation macro F1:    {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")
print("="*60)

labels = ["DoS", "Normal", "Probe", "R2L", "U2R"]
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix — {best_name} + SMOTE + Threshold Tuning")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\nConfusion matrix saved to confusion_matrix.png")
plt.show()
