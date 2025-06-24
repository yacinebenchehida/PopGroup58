import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt

# Load your DataFrame
df = pd.read_excel("Abstracts.xlsx", sheet_name="Talks")  # Adjust as needed

# Extract abstracts and drop any NaN values
abstracts = df['What is your talk abstract?'].dropna()

# Load pretrained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings for a sentence
def get_bert_embedding(abstract):
    # Tokenize the abstract and get the input IDs and attention mask
    inputs = tokenizer(abstract, return_tensors='pt', truncation=True, padding=True, max_length=512)
    
    # Get the embeddings from BERT
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use the embeddings of the [CLS] token (first token)
    return outputs.last_hidden_state[:, 0, :].numpy()

# Get BERT embeddings for all abstracts
abstract_embeddings = np.array([get_bert_embedding(abstract) for abstract in abstracts])

# Flatten the embeddings array to a 2D array (number of abstracts x embedding dimension)
abstract_embeddings = abstract_embeddings.reshape(len(abstracts), -1)

# K-means clustering on original BERT embeddings
kmeans_bert = KMeans(n_clusters=12, random_state=42)
kmeans_labels_bert = kmeans_bert.fit_predict(abstract_embeddings)

# Perform PCA for dimensionality reduction on the original embeddings
pca = PCA(n_components=2)  # Reduce to 2 dimensions for clustering
pca_result = pca.fit_transform(abstract_embeddings)

# K-means clustering directly on PCA results
kmeans_pca = KMeans(n_clusters=12, random_state=42)
kmeans_labels_pca = kmeans_pca.fit_predict(pca_result)  # Apply K-means clustering on PCA results

# Plot PCA with K-means clusters using a rainbow colormap for PCA results
fig, ax = plt.subplots(figsize=(8, 6))
scatter_pca = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans_labels_pca, cmap='rainbow', edgecolors='k')

plt.legend(*scatter_pca.legend_elements(), title="Clusters")
plt.title("PCA Plot with K-means Clusters (PCA-based)")
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

# Save the PCA plot
fig.savefig("PCA_KMeans_Clusters_rainbow.png", format='png', dpi=300)
plt.show()

# Plot K-means clusters on original BERT embeddings using a rainbow colormap
fig, ax = plt.subplots(figsize=(8, 6))
scatter_bert = ax.scatter(abstract_embeddings[:, 0], abstract_embeddings[:, 1], c=kmeans_labels_bert, cmap='rainbow', edgecolors='k')

plt.legend(*scatter_bert.legend_elements(), title="Clusters")
plt.title("K-means Clustering on BERT Embeddings")
plt.xlabel('BERT Embedding Dimension 1')
plt.ylabel('BERT Embedding Dimension 2')

# Save the BERT plot
fig.savefig("BERT_KMeans_Clusters_rainbow.png", format='png', dpi=300)
plt.show()

# TF-IDF Vectorization for keyword extraction
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(abstracts)

# Save clustering results with top keywords for both methods
def save_clustering_results(labels, method_name, abstracts, vectorizer, X):
    feature_names = vectorizer.get_feature_names_out()  # Words from the TF-IDF vectorizer

    with open(f'clustering_results_{method_name}.txt', 'w') as f:
        for label in np.unique(labels):  # Iterate over unique cluster labels
            f.write(f"Cluster {label}:\n\n")
            cluster_indices = np.where(labels == label)[0]
            
            # Write abstracts in the current cluster
            for idx in cluster_indices:
                f.write(f"Abstract {idx + 1}:\n{abstracts.iloc[idx]}\n\n")
            
            # Compute average TF-IDF scores for terms in the cluster
            cluster_tfidf = X[cluster_indices].mean(axis=0)
            cluster_tfidf = np.asarray(cluster_tfidf).flatten()  # Convert to 1D array
            
            # Get top 10 keywords for the cluster based on TF-IDF scores
            top_keywords_indices = np.argsort(cluster_tfidf)[::-1][:10]
            top_keywords = [feature_names[i] for i in top_keywords_indices]
            
            # Write top words (keywords that best define the cluster)
            f.write(f"Top words:\n{', '.join(top_keywords)}\n\n\n")

# Save K-means clusters with top keywords for BERT-based clustering
save_clustering_results(kmeans_labels_bert, "bert", abstracts, vectorizer, X)

# Save K-means clusters with top keywords for PCA-based clustering
save_clustering_results(kmeans_labels_pca, "pca", abstracts, vectorizer, X)
