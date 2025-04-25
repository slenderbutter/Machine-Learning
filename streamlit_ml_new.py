import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.impute import KNNImputer

# Load your data
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path, delimiter='\t')
    return data

# Display the file path input for the user
file_path = st.text_input("Enter the file path of the dataset:", r"C:\Degree Year 2 Semester 3\Machine Learning\Assignment\marketing_campaign.csv")

# Load data when file path is provided
if file_path:
    data = load_data(file_path)
    st.write(f"Data loaded from: {file_path}")

    # Display column names and data types to inspect the dataset
    st.write("Data Types of Columns:")
    st.write(data.dtypes)

    # Display first few rows of the data for inspection
    st.write("First few rows of the dataset:")
    st.write(data.head())

    # Preprocess the data
    def preprocess_data(data):
        # Convert all columns to numeric, coercing errors to NaN
        for column in data.columns:
            data[column] = pd.to_numeric(data[column], errors='coerce')
        
        # Drop columns that are all NaN (if any)
        data.dropna(axis=1, how='all', inplace=True)
        
        # Check and display the numeric columns after conversion
        numeric_data = data.select_dtypes(include=[np.number])
        st.write("Numeric Columns After Conversion to Numeric:")
        st.write(numeric_data.columns.tolist())

        # If there are no numeric columns, raise an error
        if numeric_data.empty:
            st.error("No numeric columns available after conversion.")
            return None, None

        # Apply KNN imputation
        imputer = KNNImputer(n_neighbors=5)
        imputed_data = imputer.fit_transform(numeric_data)

        # Convert the imputed data back to a DataFrame
        imputed_data_df = pd.DataFrame(imputed_data, columns=numeric_data.columns)

        # Replace the original data with the imputed numeric data
        data = imputed_data_df

        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        return scaled_data, data

    try:
        scaled_data, processed_data = preprocess_data(data.copy())

        if scaled_data is None or processed_data is None:
            st.warning("Data preprocessing failed. Please check your dataset for non-numeric values.")
        else:
            # Apply PCA
            pca = PCA(n_components=2)
            pca_components = pca.fit_transform(scaled_data)

            # Adding PCA components to the dataframe for visualization
            processed_data['PCA1'] = pca_components[:, 0]
            processed_data['PCA2'] = pca_components[:, 1]

            # Streamlit user interface
            st.title("📊Clustering Algorithms Dashboard")

            # Sidebar for selecting clustering method
            clustering_method = st.sidebar.selectbox("Choose Clustering Method", ["DBSCAN", "Agglomerative"])

            if clustering_method == "DBSCAN":
                # DBSCAN clustering
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                clusters = dbscan.fit_predict(processed_data[['PCA1', 'PCA2']])
                processed_data['Cluster'] = clusters
                st.write("DBSCAN Clustering")
            elif clustering_method == "Agglomerative":
                # Agglomerative clustering
                from sklearn.cluster import AgglomerativeClustering
                agglomerative = AgglomerativeClustering(n_clusters=3)
                clusters = agglomerative.fit_predict(processed_data[['PCA1', 'PCA2']])
                processed_data['Cluster'] = clusters
                st.write("Agglomerative Clustering")

            # Displaying the clusters in the scatter plot
            st.write("PCA Components Visualization with Clusters")
            fig, ax = plt.subplots()
            sns.scatterplot(x=processed_data['PCA1'], y=processed_data['PCA2'], hue=processed_data['Cluster'], palette='viridis', ax=ax)
            ax.set_title('PCA - Clusters')
            st.pyplot(fig)

            # Display silhouette score
            silhouette_avg = silhouette_score(scaled_data, processed_data['Cluster'])
            st.write(f"Silhouette Score: {silhouette_avg:.3f}")

            # Option to download the data
            st.write("Download the data with cluster assignments:")
            st.download_button("Download CSV", processed_data.to_csv(), file_name='clustered_data.csv', mime='text/csv')

    except ValueError as e:
        st.error(f"Error: {e}")
else:
    st.warning("Please provide a valid file path to load the dataset.")