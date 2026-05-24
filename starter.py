import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ==============================================================
# 1. LOAD DATA
# ==============================================================

train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
test_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

columns = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "class",
    "level",
]

df_train = pd.read_csv(train_url, names=columns)
df_test = pd.read_csv(test_url, names=columns)

# Drop difficulty level column (not a feature)
df_train = df_train.drop(columns=["level"])
df_test = df_test.drop(columns=["level"])


# ==============================================================
# 2. ENCODE CATEGORICAL FEATURES
# ==============================================================

# Merge temporarily to ensure consistent encoding across train and test
df_full = pd.concat([df_train, df_test])

# Encode categorical columns as integers
cat_cols = ["protocol_type", "service", "flag"]
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df_full[col] = le.fit_transform(df_full[col])
    label_encoders[col] = le

# ==============================================================
# 3. MAP ATTACKS TO 5 CATEGORIES
# ==============================================================

category_map = {
    "normal": "Normal",
    # DoS
    "neptune": "DoS",
    "back": "DoS",
    "land": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "mailbomb": "DoS",
    "apache2": "DoS",
    "processtable": "DoS",
    "udpstorm": "DoS",
    "worm": "DoS",
    # Probe
    "satan": "Probe",
    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "mscan": "Probe",
    "saint": "Probe",
    # R2L
    "warezclient": "R2L",
    "guess_passwd": "R2L",
    "ftp_write": "R2L",
    "imap": "R2L",
    "phf": "R2L",
    "multihop": "R2L",
    "warezmaster": "R2L",
    "spy": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    "snmpguess": "R2L",
    "snmpgetattack": "R2L",
    "httptunnel": "R2L",
    "sendmail": "R2L",
    "named": "R2L",
    # U2R
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "rootkit": "U2R",
    "perl": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R",
    "ps": "U2R",
}

df_full["category"] = df_full["class"].map(category_map).fillna("Other")

# ==============================================================
# 4. PREPARE FEATURES AND LABELS
# ==============================================================

# Drop constant column and original class labels
df_full = df_full.drop(columns=["num_outbound_cmds", "class"])

# Split back into train and test
train_len = len(df_train)
df_train_processed = df_full.iloc[:train_len].copy()
df_test_processed = df_full.iloc[train_len:].copy()

X_train = df_train_processed.drop(columns=["category"])
X_test = df_test_processed.drop(columns=["category"])

label_encoder_y = LabelEncoder()

y_train = label_encoder_y.fit_transform(df_train_processed["category"])
y_test = label_encoder_y.transform(df_test_processed["category"])


# ==============================================================
# 5. HANDLE CLASS IMBALANCE (SMOTE)
# ==============================================================


smote = SMOTE(k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)


# ==============================================================
# 6. MODEL: XGBOOST
# ==============================================================


model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="mlogloss",
)

# ==============================================================
# 7. CROSS-VALIDATION (MANDATORY)
# ==============================================================


cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=5, scoring="f1_macro")


# ==============================================================
# 8. TRAIN FINAL MODEL
# ==============================================================

model.fit(X_train_res, y_train_res)

# ==============================================================
# 9. PREDICT AND EVALUATE
# ==============================================================


y_pred = model.predict(X_test)

macro_f1 = f1_score(y_test, y_pred, average="macro")


joblib.dump(model, "xgb_model.pkl")
joblib.dump(label_encoder_y, "label_encoder.pkl")
joblib.dump(y_test, "y_test.pkl")
joblib.dump(y_pred, "y_pred.pkl")
joblib.dump(label_encoder_y, "label_encoder.pkl")
