import pandas as pd
import numpy as np
import re
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk import pos_tag
import scipy.cluster.hierarchy as sch

# Download the NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

# Load your DataFrame (assuming it's already loaded as `df`)
df = pd.read_excel("Abstracts.xlsx", sheet_name="Talks")  # adjust as needed

# Extract abstracts and drop any NaN values
abstracts = df['What is your talk abstract?'].dropna()

# Create a list of custom unwanted words (can be expanded as needed)
unwanted_words = ["previous", "highly", "identified", "potential", "high", "increased", "using", "the", "and", 
"of", "work", "panel", "little", "including", "used", "analysed", "study", "studies", "remains", "key", "understanding", 
"new", "talk", "presence", "level", "insight", "populations", "genes", "insight"]

# Get the NLTK stopwords
nltk_stopwords = stopwords.words('english')

# Combine NLTK stopwords with custom unwanted words
all_stopwords = set(nltk_stopwords + unwanted_words)

# Clean and preprocess the abstracts
def clean_text(text):
    # Remove non-alphabetical characters and make everything lowercase
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove punctuation and numbers
    text = text.lower()  # Make everything lowercase
    return text

def preprocess_text(text):
    # List of exception words that should not have their plural form changed
    exceptions = ["locus", "process", "across", "fitness", "genetic"]

    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Perform POS tagging to filter verbs
    tagged_tokens = pos_tag(tokens)

    tokens = [
        word for word, tag in tagged_tokens 
        if (
            word in exceptions  # Always keep words in the inclusion list
            or (
                word not in all_stopwords  # Exclude NLTK stopwords and custom unwanted words
                and tag not in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # Verbs
                                'JJ', 'JJR', 'JJS',  # Adjectives
                                'RB', 'RBR', 'RBS']  # Adverbs
            )
        )
    ]
    return " ".join(tokens)

# Apply cleaning and preprocessing to each abstract
cleaned_abstracts = abstracts.apply(clean_text).apply(preprocess_text)

# Vectorize the cleaned text using TF-IDF (this also removes stopwords by default)
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(cleaned_abstracts)

# Perform K-means clustering (4 clusters)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X.toarray())

# Perform Agglomerative Clustering (Hierarchical Clustering)
agg = AgglomerativeClustering(n_clusters=4)
agg_labels = agg.fit_predict(X.toarray())

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=1.1, min_samples=2)
dbscan_labels = dbscan.fit_predict(X)
print(dbscan_labels)

# Create a helper function for saving the plots
def save_plot(fig, filename):
    fig.savefig(filename, format='png', dpi=300)
    print(f"Saved plot as {filename}")

# Plot PCA with K-means clusters
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans_labels, cmap='viridis', edgecolors='k')

plt.legend(*scatter.legend_elements(), title="Clusters")
plt.title("PCA Plot with K-means Clusters")
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
save_plot(fig, "PCA_KMeans_Clusters.png")

# Plot PCA with Agglomerative Clustering (Hierarchical Clustering) clusters
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=agg_labels, cmap='viridis', edgecolors='k')

plt.legend(*scatter.legend_elements(), title="Clusters")
plt.title("PCA Plot with Agglomerative Clustering")
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
save_plot(fig, "PCA_Agglomerative_Clusters.png")

# Plot PCA with DBSCAN clusters
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=dbscan_labels, cmap='viridis', edgecolors='k')

# Highlight noise points with a separate color
scatter = ax.scatter(pca_result[dbscan_labels == -1, 0], pca_result[dbscan_labels == -1, 1], color='red', label='Noise')

plt.legend(*scatter.legend_elements(), title="Clusters")
plt.title("PCA Plot with DBSCAN Clusters")
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
save_plot(fig, "PCA_DBSCAN_Clusters.png")

# Create Dendrogram for Agglomerative Clustering (Hierarchical Clustering)
plt.figure(figsize=(10, 7))
sch.dendrogram(sch.linkage(X.toarray(), method='ward'))
plt.title('Dendrogram for Agglomerative Clustering')
plt.xlabel('Samples')
plt.ylabel('Euclidean Distance')
save_plot(plt, "Dendrogram.png")

# Save clustering results with abstracts and top words for each cluster
def save_clustering_results(labels, method_name, vectorizer, abstracts, X):
    feature_names = vectorizer.get_feature_names_out()  # Words from the TF-IDF vectorizer

    with open(f'clustering_{method_name}.txt', 'w') as f:
        for label in np.unique(labels):  # Iterate over unique cluster labels
            if label == -1:
                f.write("Cluster -1 (Noise):\n\n")
            else:
                f.write(f"Cluster {label}:\n\n")
            
            # Get indices of samples in the current cluster
            cluster_indices = np.where(labels == label)[0]
            cluster_abstracts = abstracts.iloc[cluster_indices]
            
            # Write abstracts in the current cluster
            for idx, abstract in enumerate(cluster_abstracts, 1):
                f.write(f"Abstract {idx}:\n{abstract}\n\n")
            
            # Compute average TF-IDF scores for terms in the cluster
            cluster_tfidf = X[cluster_indices].mean(axis=0)
            cluster_tfidf = np.asarray(cluster_tfidf).flatten()  # Convert to 1D array
            
            # Get top 10 keywords for the cluster
            top_keywords_indices = np.argsort(cluster_tfidf)[::-1][:10]
            top_keywords = [feature_names[i] for i in top_keywords_indices]
            
            # Write top words
            f.write(f"Top words:\n{', '.join(top_keywords)}\n\n\n")
# Save K-means clusters
save_clustering_results(kmeans_labels, "kmeans", vectorizer, abstracts, X)

# Save Agglomerative Clustering (Hierarchical) clusters
save_clustering_results(agg_labels, "agg", vectorizer, abstracts, X)

# Save DBSCAN clusters (including noise points labeled as -1)
save_clustering_results(dbscan_labels, "dbscan", vectorizer, abstracts, X)
