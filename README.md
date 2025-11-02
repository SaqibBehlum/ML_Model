# 🤖 Machine Learning Model Explorer

---

### 📝 Overview

An **interactive Machine Learning app** built with **Streamlit**. Upload datasets, train models, visualize results, and explore **supervised and unsupervised algorithms**. Ideal for **students, beginners, or hackathon demos**.

---

## 🚀 Features

---

### 📂 Dataset Upload & Preprocessing

* Upload CSV datasets and preview sample rows
* Automatic handling of missing values
* Categorical encoding & numeric feature scaling

### 🧠 Supervised Learning

* Algorithms: Decision Tree, Random Forest, SVM
* Train-test split and model training
* Displays **Accuracy**, **Classification Report**, and **Confusion Matrix**
* Feature importance visualization

### 🌀 Unsupervised Learning

* Algorithms: KMeans, Agglomerative Clustering, DBSCAN
* Cluster scatter plots
* Silhouette Score evaluation

### 📊 Automatic EDA

* Dataset shape, data types, and statistical summary
* Correlation heatmap visualization

### 🎨 Interactive UI

* Sidebar controls for learning type, dataset upload, model selection, and parameter tuning
* Clean display area for results and plots

---

## 🛠 Installation

---

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd <repo-folder>
```

2. **Create virtual environment (optional)**

```bash
python -m venv venv
# Activate environment
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App

---

```bash
streamlit run app.py
```

* Open in browser at `https://mlmodel-fwm2x6sgpxfngjr79fspyn.streamlit.app/`
* On **Streamlit Cloud or Hugging Face**, follow their instructions to deploy

---

## 📋 Requirements

---

```txt
streamlit==1.37.0
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

## 💡 Usage

---

1. Select **learning type**: Supervised / Unsupervised / Auto EDA
2. Upload a **CSV dataset**
3. For **Supervised Learning**, select target column & algorithm
4. For **Unsupervised Learning**, select clustering algorithm & parameters
5. Click **Train / Run Model**
6. View metrics, reports, confusion matrix, cluster plots, and feature importance

---

## ⚠️ Notes

---

* Target column must be categorical for classification tasks
* Large datasets may exceed memory limits on free hosting
* Confusion matrix & cluster visuals display after training

---

## 👤 Author

---

**Saqib Ahmed** – Hackathon
