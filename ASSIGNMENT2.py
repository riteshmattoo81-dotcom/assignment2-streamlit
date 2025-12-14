import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

plt.rcParams['figure.figsize'] = (10, 5)

# -----------------------
# Helper functions
# -----------------------
def create_models():
    return {
        "Logistic regression": LogisticRegression(max_iter=1000),
        "L1 Logistic (Lasso)": LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

def evaluate_models(X_train, X_test, y_train, y_test):
    models = create_models()
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, pos_label=1)
        f1  = f1_score(y_test, y_pred, pos_label=1)
        results[name] = (acc, rec, f1)
    return results, models

def get_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).ravel()
    else:
        raise ValueError("Model does not provide feature importance.")
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)

# -----------------------
# Streamlit app
# -----------------------
def main():
    # Top title like your reference screenshot
    st.title("Supply Chain Fraud & Late Delivery Dashboard")
    st.subheader("Model Performance Comparison")

    # ---- Data input ----
    st.sidebar.header("Data input")
    upload = st.sidebar.file_uploader("Upload DataCoSupplyChainDataset.csv", type=["csv"])

    if upload is not None:
        df = pd.read_csv(upload, encoding="latin1")
    else:
        st.sidebar.info("Using local 'DataCoSupplyChainDataset.csv' if available.")
        try:
            df = pd.read_csv("DataCoSupplyChainDataset.csv", encoding="latin1")
        except FileNotFoundError:
            st.error("No file uploaded and 'DataCoSupplyChainDataset.csv' not found in working directory.")
            return

    st.write("Loaded data shape:", df.shape)

    # ---- Targets ----
    df["fraud"] = np.where(df.get("Order Status") == "SUSPECTED_FRAUD", 1, 0)
    df["late_delivery"] = np.where(df.get("Delivery Status") == "Late delivery", 1, 0)

    # ---- Cleaning ----
    df_clean = df.copy(deep=True)

    cols_to_drop = [
        "Order Status", "Delivery Status",
        "Order Id", "Customer Id", "Customer Email",
        "Product Name", "Order Item Id"
    ]
    cols_to_drop = [c for c in cols_to_drop if c in df_clean.columns]
    df_clean = df_clean.drop(columns=cols_to_drop)

    all_null_cols = df_clean.columns[df_clean.isna().all()]
    if len(all_null_cols) > 0:
        st.write("Dropping all-NaN columns:", list(all_null_cols))
        df_clean = df_clean.drop(columns=all_null_cols)

    for col in df_clean.columns:
        if df_clean[col].dtype == "O":
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        else:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    st.write("Total remaining NaNs:", int(df_clean.isna().sum().sum()))

    cat_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])

    # ---- Build features ----
    target_cols = ["fraud", "late_delivery"]
    feature_cols = [c for c in df_clean.columns if c not in target_cols]

    X = df_clean[feature_cols].values
    y_fraud = df_clean["fraud"].values
    y_late = df_clean["late_delivery"].values

    var_sel = VarianceThreshold(threshold=0.0)
    X = var_sel.fit_transform(X)
    feature_cols = np.array(feature_cols)[var_sel.get_support()]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.write("Feature matrix shape after scaling:", X_scaled.shape)

    # ---- Compare splits and build model performance table ----
    splits = {"80-20": 0.2, "70-30": 0.3}
    all_results = {}

    for split_name, test_size in splits.items():
        # FRAUD
        X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
            X_scaled, y_fraud, test_size=test_size, random_state=42, stratify=y_fraud
        )
        fraud_res, _ = evaluate_models(X_train_f, X_test_f, y_train_f, y_test_f)

        # LATE
        X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
            X_scaled, y_late, test_size=test_size, random_state=42, stratify=y_late
        )
        late_res, _ = evaluate_models(X_train_l, X_test_l, y_train_l, y_test_l)

        rows = []
        model_names = list(create_models().keys())
        for name in model_names:
            acc_f, rec_f, f1_f = fraud_res[name]
            acc_l, rec_l, f1_l = late_res[name]
            rows.append([name, acc_f, rec_f, f1_f, acc_l, rec_l, f1_l])

        cols = [
            "Model",
            "Accuracy (Fraud)", "Recall (Fraud)", "F1 (Fraud)",
            "Accuracy (Late)",  "Recall (Late)",  "F1 (Late)"
        ]
        results_df = pd.DataFrame(rows, columns=cols)

        all_results[split_name] = {
            "df": results_df,
            "fraud": (X_train_f, X_test_f, y_train_f, y_test_f),
            "late":  (X_train_l, X_test_l, y_train_l, y_test_l)
        }

    # choose best split by average F1
    best_split = None
    best_score = -np.inf
    for split_name, obj in all_results.items():
        df_res = obj["df"]
        mean_f1 = (df_res["F1 (Fraud)"].mean() + df_res["F1 (Late)"].mean()) / 2.0
        if mean_f1 > best_score:
            best_score = mean_f1
            best_split = split_name

    best_results_df = all_results[best_split]["df"]

    # Top comparison table (like screenshot)
    st.caption(f"Best split: {best_split} (mean F1 = {best_score:.4f})")
    st.dataframe(best_results_df, use_container_width=True)

    # ---- Train best model for each target ----
    X_train_f, X_test_f, y_train_f, y_test_f = all_results[best_split]["fraud"]
    X_train_l, X_test_l, y_train_l, y_test_l = all_results[best_split]["late"]

    fraud_res, fraud_models = evaluate_models(X_train_f, X_test_f, y_train_f, y_test_f)
    late_res, late_models  = evaluate_models(X_train_l, X_test_l, y_train_l, y_test_l)

    best_model_fraud_name = max(fraud_res, key=lambda k: fraud_res[k][2])
    best_model_late_name  = max(late_res,  key=lambda k: late_res[k][2])

    st.markdown("---")
    st.subheader("Best models for each target")
    st.write("Best model for FRAUD:", best_model_fraud_name, "metrics:", fraud_res[best_model_fraud_name])
    st.write("Best model for LATE DELIVERY:", best_model_late_name, "metrics:", late_res[best_model_late_name])

    best_model_fraud = fraud_models[best_model_fraud_name]
    best_model_late  = late_models[best_model_late_name]

    # ---- Feature importance and plots ----
    st.markdown("---")
    st.subheader("Top 5 feature importances")

    # FRAUD
    fi_fraud = get_feature_importance(best_model_fraud, feature_cols)
    top5_fraud = fi_fraud.head(5)
    st.write("Top 5 features for FRAUD detection:")
    st.dataframe(top5_fraud.to_frame("importance"))

    fig1, ax1 = plt.subplots()
    top5_fraud.sort_values().plot(kind="barh", ax=ax1)
    ax1.set_title(f"Top 5 Feature Importances - Fraud ({best_model_fraud_name})")
    ax1.set_xlabel("Importance")
    st.pyplot(fig1)

    # LATE DELIVERY
    fi_late = get_feature_importance(best_model_late, feature_cols)
    top5_late = fi_late.head(5)
    st.write("Top 5 features for LATE DELIVERY prediction:")
    st.dataframe(top5_late.to_frame("importance"))

    fig2, ax2 = plt.subplots()
    top5_late.sort_values().plot(kind="barh", ax=ax2)
    ax2.set_title(f"Top 5 Feature Importances - Late Delivery ({best_model_late_name})")
    ax2.set_xlabel("Importance")
    st.pyplot(fig2)

    # ---- Download best split table ----
    st.subheader("Download best split results")
    csv = best_results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Table-4 style CSV",
        data=csv,
        file_name="table4_best_split.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
st.caption(f"Best split: {best_split} (mean F1 = {best_score:.4f})")
st.dataframe(best_results_df, use_container_width=True)
