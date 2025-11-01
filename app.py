import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="ML Model Explorer", page_icon="🤖", layout="wide")
st.title("🤖 Machine Learning Model Explorer")
st.markdown("### Explore, Train, Compare, and Visualize Machine Learning Models Instantly!")

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.title("⚙️ Controls")
learning_type = st.sidebar.radio("Learning Type", ("Supervised", "Unsupervised", "Auto EDA"))
st.sidebar.markdown("---")
sidebar_upload = st.sidebar.file_uploader("📂 Upload Dataset (CSV)", type=["csv"])

# -----------------------------
# MAIN UPLOAD OPTION
# -----------------------------
st.markdown("### 📁 Upload Dataset (You can upload here or in the sidebar)")
main_upload = st.file_uploader("Upload CSV file", type=["csv"], key="main_upload")

uploaded_file = main_upload if main_upload is not None else sidebar_upload

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    df = df.dropna()
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # -----------------------------
    # AUTO EDA SECTION
    # -----------------------------
    if learning_type == "Auto EDA":
        st.header("📊 Automatic Exploratory Data Analysis (EDA)")
        st.markdown("Get instant insights into your dataset before model training.")

        col1, col2 = st.columns(2)
        with col1:
            st.write("### 🔢 Dataset Shape")
            st.write(df.shape)
            st.write("### 🧱 Column Names")
            st.write(list(df.columns))
            st.write("### 📋 Data Types")
            st.write(df.dtypes)

        with col2:
            st.write("### 📈 Statistical Summary")
            st.dataframe(df.describe())

        st.write("### 🔍 Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.write("### 📊 Distribution of Each Feature")
        for column in df.columns[:4]:  # show first 4 for simplicity
            fig, ax = plt.subplots()
            sns.histplot(df[column], kde=True, ax=ax)
            st.pyplot(fig)

    # -----------------------------
    # SUPERVISED LEARNING
    # -----------------------------
    elif learning_type == "Supervised":
        st.sidebar.markdown("### 🎯 Supervised Learning")
        target_col = st.sidebar.selectbox("Select Target Column", df.columns)

        model_mode = st.sidebar.radio("Mode", ("Single Model", "Compare Models"))

        X = df_scaled.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_mode == "Single Model":
            model_choice = st.sidebar.selectbox(
                "Select Model",
                ["Decision Tree Classifier", "Random Forest Classifier", "Support Vector Machine (SVM)"]
            )

            if model_choice == "Decision Tree Classifier":
                max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
                model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

            elif model_choice == "Random Forest Classifier":
                n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
                model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

            elif model_choice == "Support Vector Machine (SVM)":
                c_val = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
                kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"])
                model = SVC(C=c_val, kernel=kernel)

            if st.sidebar.button("🚀 Train Model"):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.success(f"✅ Model trained successfully! Accuracy: **{acc:.2f}**")

                st.subheader("📉 Confusion Matrix")
                fig, ax = plt.subplots()
                sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt="d", ax=ax)
                st.pyplot(fig)

                st.subheader("📋 Classification Report")
                st.text(classification_report(y_test, y_pred))

                # Feature Importance (if available)
                if hasattr(model, "feature_importances_"):
                    st.subheader("📊 Feature Importance")
                    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                    st.bar_chart(importance.head(10))

        else:  # Compare Models
            st.subheader("🤝 Model Comparison Dashboard")
            models = {
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "SVM": SVC()
            }
            results = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                results[name] = accuracy_score(y_test, preds)

            st.bar_chart(pd.Series(results, name="Accuracy"))
            st.write("### 📋 Model Accuracies")
            st.dataframe(pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"]))

    # -----------------------------
    # UNSUPERVISED LEARNING
    # -----------------------------
    elif learning_type == "Unsupervised":
        st.sidebar.markdown("### 🧠 Unsupervised Learning")
        model_choice = st.sidebar.selectbox(
            "Select Model",
            ["KMeans", "Agglomerative Clustering", "DBSCAN"]
        )

        if model_choice == "KMeans":
            k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)
            model = KMeans(n_clusters=k, random_state=42)

        elif model_choice == "Agglomerative Clustering":
            n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
            model = AgglomerativeClustering(n_clusters=n_clusters)

        elif model_choice == "DBSCAN":
            eps = st.sidebar.slider("Epsilon (eps)", 0.1, 5.0, 0.5)
            min_samples = st.sidebar.slider("Min Samples", 2, 20, 5)
            model = DBSCAN(eps=eps, min_samples=min_samples)

        if st.sidebar.button("🌀 Run Clustering"):
            clusters = model.fit_predict(df_scaled)
            df_scaled["Cluster"] = clusters
            st.success("✅ Clustering Completed!")

            st.subheader("🎨 Cluster Visualization")
            fig, ax = plt.subplots()
            sns.scatterplot(
                x=df_scaled.iloc[:, 0],
                y=df_scaled.iloc[:, 1],
                hue="Cluster",
                palette="tab10",
                data=df_scaled,
                ax=ax
            )
            st.pyplot(fig)

            if len(set(clusters)) > 1 and -1 not in clusters:
                score = silhouette_score(df_scaled.drop(columns=["Cluster"]), clusters)
                st.info(f"Silhouette Score: **{score:.2f}**")
            else:
                st.warning("⚠️ Silhouette score not available (only one cluster detected).")

            st.subheader("📊 Clustered Data Preview")
            st.dataframe(df_scaled.head())

else:
    st.info("👈 Upload a dataset to get started — or try the Auto EDA option first!")

# Footer
st.markdown("<hr><center>Built with ❤️ by Saqib Ahmed | Hackathon Edition</center>", unsafe_allow_html=True)
