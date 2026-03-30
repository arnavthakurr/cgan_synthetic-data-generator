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
    page_icon="AI",
    layout="wide",
)


def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 28%),
                radial-gradient(circle at top right, rgba(180, 83, 9, 0.10), transparent 22%),
                linear-gradient(180deg, #f8faf7 0%, #eef4f1 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        .hero {
            background: linear-gradient(135deg, #0f766e 0%, #164e63 100%);
            color: #f8fafc;
            border-radius: 24px;
            padding: 1.4rem 1.5rem;
            box-shadow: 0 18px 60px rgba(15, 23, 42, 0.18);
            margin-bottom: 1.25rem;
        }
        .hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2.1rem;
            line-height: 1.1;
        }
        .hero p {
            margin: 0;
            color: rgba(248, 250, 252, 0.92);
            font-size: 1rem;
        }
        .mini-card {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(15, 118, 110, 0.12);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.07);
            backdrop-filter: blur(6px);
            margin-bottom: 0.9rem;
        }
        .mini-card .label {
            color: #486581;
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.35rem;
        }
        .mini-card .value {
            color: #102a43;
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.1;
        }
        .section-card {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(15, 118, 110, 0.10);
            border-radius: 22px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 14px 36px rgba(15, 23, 42, 0.07);
            margin-bottom: 1rem;
        }
        .section-title {
            color: #102a43;
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }
        .muted {
            color: #486581;
            font-size: 0.95rem;
        }
        .drift-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95rem;
            overflow: hidden;
            border-radius: 16px;
        }
        .drift-table thead th {
            background: #d9ece8;
            color: #102a43;
            text-align: left;
            padding: 0.8rem;
        }
        .drift-table tbody td {
            padding: 0.78rem 0.8rem;
            border-top: 1px solid #e6eeeb;
        }
        .drift-table tbody tr:nth-child(even) {
            background: #f8fbfa;
        }
        .pill {
            display: inline-block;
            background: #e6fffa;
            color: #0f766e;
            border: 1px solid rgba(15, 118, 110, 0.18);
            padding: 0.22rem 0.55rem;
            border-radius: 999px;
            font-size: 0.8rem;
            margin-right: 0.45rem;
            margin-bottom: 0.35rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #edf6f3 0%, #e3eeeb 100%);
            border-right: 1px solid rgba(15, 118, 110, 0.10);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.45rem;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.68);
            border-radius: 999px;
            padding: 0.45rem 1rem;
        }
        .stDownloadButton button, .stButton button {
            border-radius: 999px;
            font-weight: 600;
        }
        @media (max-width: 900px) {
            .hero h1 {
                font-size: 1.65rem;
            }
            .block-container {
                padding-top: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
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


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dataset_summary(name, df):
    fraud_rate = df["isFraud"].mean() * 100 if "isFraud" in df.columns else 0
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card(f"{name} rows", f"{len(df):,}")
    with c2:
        metric_card(f"{name} columns", len(df.columns))
    with c3:
        metric_card(f"{name} fraud rate", f"{fraud_rate:.2f}%")


def render_distribution_plot(real_df, synthetic_df, feature):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    sns.kdeplot(real_df[feature], label="Real", fill=True, ax=ax, color="#0f766e")
    sns.kdeplot(
        synthetic_df[feature],
        label="Synthetic",
        fill=True,
        ax=ax,
        color="#c2410c",
    )
    ax.set_title(f"Feature Distribution: {feature}", fontsize=14, pad=12)
    ax.set_xlabel(feature)
    ax.set_ylabel("Density")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    st.pyplot(fig, clear_figure=True)


def build_drift_df(real_df, synthetic_df, numeric_features):
    rows = []
    for feature in numeric_features:
        real_mean = real_df[feature].mean()
        synthetic_mean = synthetic_df[feature].mean()
        rows.append(
            {
                "feature": str(feature),
                "real_mean": round(real_mean, 6),
                "synthetic_mean": round(synthetic_mean, 6),
                "abs_mean_gap": round(abs(real_mean - synthetic_mean), 6),
                "real_std": round(real_df[feature].std(), 6),
                "synthetic_std": round(synthetic_df[feature].std(), 6),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_mean_gap", ascending=False)


def render_drift_table(drift_df):
    display_df = drift_df.head(12).copy()
    html = display_df.to_html(
        index=False,
        classes="drift-table",
        border=0,
        justify="left",
    )
    st.markdown(html, unsafe_allow_html=True)
    st.download_button(
        "Download drift report",
        data=drift_df.to_csv(index=False),
        file_name="feature_drift_report.csv",
        mime="text/csv",
    )


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


inject_css()

st.markdown(
    """
    <div class="hero">
        <h1>CGAN Synthetic Fraud Dashboard</h1>
        <p>Compare real and synthetic transaction behavior, inspect feature drift, and run a lightweight fraud-classifier evaluation in one deployable Streamlit app.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

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
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Start By Uploading Data</div>
            <div class="muted">Use the sidebar to upload <code>processed_data.csv</code> and <code>synthetic_transactions.csv</code>. Once both files are available, the dashboard will unlock the overview, comparison, and classifier tabs.</div>
        </div>
        """,
        unsafe_allow_html=True,
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

drift_df = build_drift_df(real_df, synthetic_df, numeric_features)

top_row_1, top_row_2, top_row_3, top_row_4 = st.columns(4)
with top_row_1:
    metric_card("Shared features", len(numeric_features))
with top_row_2:
    metric_card("Largest drift", f"{drift_df.iloc[0]['abs_mean_gap']:.4f}")
with top_row_3:
    metric_card("Real rows", f"{len(real_df):,}")
with top_row_4:
    metric_card("Synthetic rows", f"{len(synthetic_df):,}")

st.markdown(
    "".join(
        [
            f'<span class="pill">Top drift: {feature}</span>'
            for feature in drift_df["feature"].head(5).tolist()
        ]
    ),
    unsafe_allow_html=True,
)

overview_tab, compare_tab, model_tab = st.tabs(
    ["Overview", "Feature Comparison", "Classifier Evaluation"]
)

with overview_tab:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="muted">Quick preview of the uploaded datasets and their label balance.</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        render_dataset_summary("Real", real_df)
        st.dataframe(real_df.head(8), use_container_width=True, height=320)
    with right:
        render_dataset_summary("Synthetic", synthetic_df)
        st.dataframe(synthetic_df.head(8), use_container_width=True, height=320)
    st.markdown("</div>", unsafe_allow_html=True)

with compare_tab:
    left, right = st.columns([1.4, 1])
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">Feature Distribution Comparison</div>',
            unsafe_allow_html=True,
        )
        selected_feature = st.selectbox("Choose a feature", numeric_features)
        render_distribution_plot(real_df, synthetic_df, selected_feature)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Mean Drift Table</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="muted">Top features ranked by absolute mean difference. This HTML table avoids the Arrow rendering bug from the previous version.</div>',
            unsafe_allow_html=True,
        )
        render_drift_table(drift_df)
        st.markdown("</div>", unsafe_allow_html=True)

with model_tab:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Random Forest Evaluation</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="muted">This mirrors the logic from <code>fraud_classifier.py</code> by training a small classifier on the combined real and synthetic dataset.</div>',
        unsafe_allow_html=True,
    )
    if st.button("Run evaluation"):
        with st.spinner("Training classifier..."):
            accuracy, report = run_classifier(real_df, synthetic_df)
        m1, m2 = st.columns([0.4, 0.6])
        with m1:
            metric_card("Accuracy", f"{accuracy:.4f}")
        with m2:
            st.code(report)
        st.download_button(
            "Download classification report",
            data=StringIO(report).getvalue(),
            file_name="classification_report.txt",
            mime="text/plain",
        )
