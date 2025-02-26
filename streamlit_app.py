import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
from sklearn.ensemble import IsolationForest
from io import StringIO
import datetime
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import numpy as np

# Configure Google API Key
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# App Title and Description
st.title("AI-Powered QA, Validation & Analytics Tool")
st.write("An all-in-one solution for advanced QA, validations, and dataset analytics using AI.")

# Sidebar: File Upload and Options
st.sidebar.header("Upload Files")
uploaded_files = st.sidebar.file_uploader("Upload your file(s)", type=["csv", "xlsx", "json"], accept_multiple_files=True)

# Initialize session state for uploaded data
if "uploaded_data" not in st.session_state:
    st.session_state["uploaded_data"] = {}

# Load Data Files
for uploaded_file in uploaded_files:
    if uploaded_file.name not in st.session_state["uploaded_data"]:
        file_type = uploaded_file.name.split(".")[-1]
        try:
            if file_type == "csv":
                df = pd.read_csv(uploaded_file)
            elif file_type == "xlsx":
                df = pd.read_excel(uploaded_file)
            elif file_type == "json":
                df = pd.read_json(uploaded_file)
            else:
                st.sidebar.error(f"Unsupported file type: {file_type}")
                continue
            st.session_state["uploaded_data"][uploaded_file.name] = df
        except Exception as e:
            st.sidebar.error(f"Error loading {uploaded_file.name}: {e}")

# Display Uploaded Files
if st.session_state["uploaded_data"]:
    st.sidebar.write("Uploaded Files:")
    for file_name in st.session_state["uploaded_data"]:
        st.sidebar.write(f"- {file_name}")

# Select Dataset for QA
if st.session_state["uploaded_data"]:
    dataset_name = st.selectbox("Select a dataset for QA:", list(st.session_state["uploaded_data"].keys()))
    if dataset_name:
        data = st.session_state["uploaded_data"][dataset_name]
        st.write(f"### Preview of {dataset_name}")
        st.dataframe(data.head(10))

        # Data Cleaning Options
        st.subheader("Data Cleaning Options")
        if st.button("Remove Duplicates"):
            data = data.drop_duplicates()
            st.success("Duplicates removed!")
            st.dataframe(data)

        # Handling Missing Values
        st.subheader("Handle Missing Values")
        imputation_strategy = st.selectbox("Choose imputation strategy", ["Mean", "Median", "Most Frequent"])
        if st.button("Impute Missing Values"):
            imputer = SimpleImputer(strategy=imputation_strategy.lower())
            data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
            st.success(f"Missing values imputed using {imputation_strategy} strategy.")
            st.dataframe(data)

        # QA Validation and Statistical Checks
        st.subheader("Run Advanced QA Validations")
        validation_query = st.text_input("Enter a validation query (e.g., 'Check for missing values in Column X'):")

        if st.button("Run Validation"):
            try:
                # Example Validations
                if "missing values" in validation_query.lower():
                    missing_count = data.isnull().sum()
                    st.write("### Missing Values Summary:")
                    st.write(missing_count)
                elif "column stats" in validation_query.lower():
                    st.write("### Column Statistics:")
                    st.write(data.describe())
                else:
                    st.warning("Validation query not recognized.")
            except Exception as e:
                st.error(f"Error during validation: {e}")

        # Anomaly Detection
        st.subheader("Anomaly Detection")
        anomaly_column = st.selectbox("Select a column for anomaly detection:", data.select_dtypes(include=["float", "int"]).columns)

        if st.button("Detect Anomalies"):
            try:
                iso_forest = IsolationForest(contamination=0.05, random_state=42)
                data["Anomaly"] = iso_forest.fit_predict(data[[anomaly_column]])
                anomalies = data[data["Anomaly"] == -1]
                st.write(f"### Detected Anomalies in {anomaly_column}:")
                st.dataframe(anomalies)
            except Exception as e:
                st.error(f"Error during anomaly detection: {e}")

        # Visualization
        st.subheader("Visualization")
        chart_column = st.selectbox("Select a column to visualize:", data.columns)
        chart_type = st.radio("Select chart type:", ["Histogram", "Line Chart", "Bar Chart", "Box Plot", "Correlation Heatmap"])

        if st.button("Generate Chart"):
            try:
                plt.figure()
                if chart_type == "Histogram":
                    plt.hist(pd.to_numeric(data[chart_column], errors='coerce').dropna(), bins=10)
                elif chart_type == "Line Chart":
                    pd.to_numeric(data[chart_column], errors='coerce').dropna().plot(kind="line")
                elif chart_type == "Bar Chart":
                    pd.to_numeric(data[chart_column], errors='coerce').dropna().value_counts().plot(kind="bar")
                elif chart_type == "Box Plot":
                    sns.boxplot(x=pd.to_numeric(data[chart_column], errors='coerce').dropna())
                elif chart_type == "Correlation Heatmap":
                    corr = data.corr()
                    sns.heatmap(corr, annot=True, cmap="coolwarm")
                plt.title(f"{chart_type} of {chart_column}")
                st.pyplot(plt)
                plt.clf()
            except Exception as e:
                st.error(f"Error generating chart: {e}")

        # AI-Powered QA with Generative AI
        st.subheader("AI-Powered QA")
        user_prompt = st.text_input("Ask a question about your dataset (e.g., 'Summarize Column A'):")

        if st.button("Ask AI"):
            try:
                # Use generative AI to respond
                model = genai.GenerativeModel("gemini-1.5-flash")
                ai_response = model.generate_content(user_prompt)
                st.write("### AI Response:")
                st.write(ai_response.text)
            except Exception as e:
                st.error(f"AI Error: {e}")

        # Feature Scaling
        st.subheader("Feature Scaling")
        scale_option = st.selectbox("Select Scaling Method", ["Standard Scaling", "Min-Max Scaling", "Robust Scaling"])
        if st.button("Scale Features"):
            try:
                scaler = None
                if scale_option == "Standard Scaling":
                    scaler = StandardScaler()
                elif scale_option == "Min-Max Scaling":
                    scaler = MinMaxScaler()
                elif scale_option == "Robust Scaling":
                    scaler = RobustScaler()

                scaled_data = pd.DataFrame(scaler.fit_transform(data.select_dtypes(include=["float", "int"])), columns=data.select_dtypes(include=["float", "int"]).columns)
                st.write(f"Scaled data using {scale_option} method:")
                st.dataframe(scaled_data)
            except Exception as e:
                st.error(f"Error during feature scaling: {e}")

        # PCA for Dimensionality Reduction
        st.subheader("PCA for Dimensionality Reduction")
        if st.button("Apply PCA"):
            try:
                numeric_data = data.select_dtypes(include=["float", "int"]).dropna()
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(numeric_data)
                st.write("PCA Components:")
                st.dataframe(pd.DataFrame(pca_result, columns=["PC1", "PC2"]))
                st.write(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
            except Exception as e:
                st.error(f"Error during PCA: {e}")

        # Feature Engineering Options
        st.subheader("Feature Engineering")
        feature_columns = st.multiselect("Select columns to engineer features from", data.columns.tolist())
        if st.button("Generate Features"):
            try:
                for col in feature_columns:
                    if np.issubdtype(data[col].dtype, np.number):
                        data[f"{col}_log"] = np.log1p(data[col].replace([np.inf, -np.inf], np.nan).dropna())
                        data[f"{col}_sqrt"] = np.sqrt(data[col].replace([np.inf, -np.inf], np.nan).dropna())
                st.write("New Features Added:")
                st.dataframe(data.head())
            except Exception as e:
                st.error(f"Error during feature engineering: {e}")

        # Correlation and Multi-collinearity
        st.subheader("Correlation Analysis")
        if st.button("Check Correlations"):
            try:
                numeric_data = data.select_dtypes(include=["float", "int"]).dropna()
                corr = numeric_data.corr()
                st.write("Correlation Matrix:")
                st.dataframe(corr)
                if corr.abs().gt(0.8).any().any():
                    st.warning("Warning: High correlation detected between some variables.")
            except Exception as e:
                st.error(f"Error during correlation analysis: {e}")

        # Scheduled Task Example
        st.subheader("Scheduled Validation")
        schedule_task = st.checkbox("Enable daily validation")
        if schedule_task:
            st.write("### Scheduled Task Example")
            st.write(f"Task Scheduled: Validations will run daily at {datetime.datetime.now().strftime('%H:%M:%S')}.")

        # Save Processed Data
        st.subheader("Save Processed Data")
        if st.button("Download Processed File"):
            try:
                buffer = StringIO()
                data.to_csv(buffer, index=False)
                st.download_button(label="Download CSV", data=buffer.getvalue(), file_name="processed_data.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Error exporting file: {e}")
