from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="CGAN Synthetic Fraud Dashboard",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_csv_from_upload(uploaded_file):
    return pd.read_csv(uploaded_file)


@st.cache_data(show_spinner=False)
def load_csv_from_path(path):
    return pd.read_csv(path)


def try_load_default_csv(path):
    try:
        return load_csv_from_path(path)
    except FileNotFoundError:
        return None


def render_dataset_summary(name, df):
    fraud_rate = df["isFraud"].mean() * 100 if "isFraud" in df.columns else None
    col1, col2, col3 = st.columns(3)
    col1.metric(f"{name} rows", f"{len(df):,}")
    col2.metric(f"{name} columns", len(df.columns))
    if fraud_rate is not None:
        col3.metric(f"{name} fraud rate", f"{fraud_rate:.2f}%")


def render_distribution_plot(real_df, synthetic_df, feature):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.kdeplot(real_df[feature], label="Real", fill=True, ax=ax)
    sns.kdeplot(synthetic_df[feature], label="Synthetic", fill=True, ax=ax)
    ax.set_title(f"Feature Distribution: {feature}")
    ax.legend()
    st.pyplot(fig, clear_figure=True)


def render_feature_drift(real_df, synthetic_df, numeric_features):
    rows = []
    for feature in numeric_features:
        rows.append(
            {
                "feature": feature,
                "real_mean": real_df[feature].mean(),
                "synthetic_mean": synthetic_df[feature].mean(),
                "abs_mean_gap": abs(real_df[feature].mean() - synthetic_df[feature].mean()),
                "real_std": real_df[feature].std(),
                "synthetic_std": synthetic_df[feature].std(),
            }
        )
    drift_df = pd.DataFrame(rows).sort_values("abs_mean_gap", ascending=False)
    st.dataframe(drift_df, use_container_width=True)


def run_classifier(real_df, synthetic_df):
    combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)
    X = combined_df.drop("isFraud", axis=1)
    y = combined_df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return accuracy_score(y_test, preds), classification_report(y_test, preds)


st.title("Synthetic Fraud Data Dashboard")
st.caption("Deployable Streamlit interface for exploring real vs synthetic transaction data.")

st.sidebar.header("Data Sources")
st.sidebar.write(
    "Upload your generated CSV files, or let the app use local defaults if they exist."
)

real_upload = st.sidebar.file_uploader(
    "Processed real dataset", type="csv", key="real_csv"
)
synthetic_upload = st.sidebar.file_uploader(
    "Synthetic dataset", type="csv", key="synthetic_csv"
)

real_df = (
    load_csv_from_upload(real_upload)
    if real_upload is not None
    else try_load_default_csv("processed_data.csv")
)
synthetic_df = (
    load_csv_from_upload(synthetic_upload)
    if synthetic_upload is not None
    else try_load_default_csv("synthetic_transactions.csv")
)

if real_df is None or synthetic_df is None:
    st.info(
        "Upload `processed_data.csv` and `synthetic_transactions.csv` from the sidebar to explore the dashboard."
    )
    st.markdown(
        """
        **Expected files**
        - `processed_data.csv`: output from `data_preprocessing.py`
        - `synthetic_transactions.csv`: output from `train.py`
        """
    )
    st.stop()

if "isFraud" not in real_df.columns or "isFraud" not in synthetic_df.columns:
    st.error("Both CSV files must include an `isFraud` column.")
    st.stop()

shared_columns = [
    column for column in real_df.columns if column in synthetic_df.columns
]
numeric_features = [
    column
    for column in shared_columns
    if column != "isFraud"
    and pd.api.types.is_numeric_dtype(real_df[column])
    and pd.api.types.is_numeric_dtype(synthetic_df[column])
]

if not numeric_features:
    st.error("No shared numeric features were found between the uploaded datasets.")
    st.stop()

overview_tab, compare_tab, model_tab = st.tabs(
    ["Overview", "Feature Comparison", "Classifier Evaluation"]
)

with overview_tab:
    st.subheader("Dataset Overview")
    left, right = st.columns(2)
    with left:
        render_dataset_summary("Real", real_df)
        st.dataframe(real_df.head(10), use_container_width=True)
    with right:
        render_dataset_summary("Synthetic", synthetic_df)
        st.dataframe(synthetic_df.head(10), use_container_width=True)

with compare_tab:
    st.subheader("Feature Distribution Comparison")
    selected_feature = st.selectbox("Choose a feature", numeric_features)
    render_distribution_plot(real_df, synthetic_df, selected_feature)

    st.subheader("Mean Drift Table")
    render_feature_drift(real_df, synthetic_df, numeric_features)

with model_tab:
    st.subheader("Random Forest Evaluation")
    st.write(
        "This reproduces the idea from `fraud_classifier.py` by training a classifier on the combined real and synthetic dataset."
    )
    if st.button("Run evaluation"):
        with st.spinner("Training classifier..."):
            accuracy, report = run_classifier(real_df, synthetic_df)
        st.metric("Accuracy", f"{accuracy:.4f}")
        st.text(report)
        st.download_button(
            "Download classification report",
            data=StringIO(report).getvalue(),
            file_name="classification_report.txt",
            mime="text/plain",
        )
