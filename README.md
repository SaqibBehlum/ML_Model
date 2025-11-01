🤖 Machine Learning Model Explorer

Project Overview
This is an interactive Machine Learning Model Explorer built with Streamlit. It allows users to upload datasets, experiment with Supervised and Unsupervised learning algorithms, and visualize results in an intuitive dashboard. Perfect for students, beginners, or hackathon demos.

🔹 Features
Dataset Handling

Upload CSV datasets and preview sample rows.

Automatic handling of missing values and categorical encoding.

Feature scaling for numeric columns.

Supervised Learning

Algorithms:

Decision Tree Classifier

Random Forest Classifier

Support Vector Machine (SVM)

Train-test split and model training.

Shows Accuracy, Classification Report, and Confusion Matrix.

Feature importance visualization (if available).

Unsupervised Learning

Algorithms:

KMeans

Agglomerative Clustering

DBSCAN

Visualizes clusters using scatter plots.

Silhouette Score calculation for clustering evaluation.

Automatic EDA

Dataset shape, data types, and statistical summary.

Correlation heatmap visualization.

Interactive UI

Sidebar controls for:

Selecting learning type (Supervised / Unsupervised / Auto EDA)

Uploading dataset

Choosing model algorithm and parameters

Training / Running models

Clean and responsive main display area for results and plots.

🔹 Installation

Clone the repository

git clone <your-repo-url>
cd <repo-folder>


Create a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows


Install dependencies

pip install -r requirements.txt

🔹 Running the App
streamlit run app.py


The app will open in your browser at http://localhost:8501.

On deployment platforms (Streamlit Cloud, Hugging Face), follow their instructions to run the app publicly.

🔹 Requirements
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn

🔹 Usage

Select learning type from the sidebar.

Upload a CSV dataset.

For Supervised Learning, select the target column and algorithm, then adjust parameters.

For Unsupervised Learning, select algorithm and parameters.

Click Train / Run Model.

View results, reports, and visualizations.

🔹 Notes

Ensure your target column is categorical for classification tasks.

Large datasets may cause memory issues on free hosting platforms.

Confusion matrix and cluster visualizations are displayed after training.

🔹 Author

Saqib Ahmed – Hackathon 
