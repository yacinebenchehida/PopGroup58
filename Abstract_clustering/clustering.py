import pandas as pd
import numpy as np
import re
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk import pos_tag

# Download the NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

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

# Perform LDA
lda = LDA(n_components=2)
lda_result = lda.fit_transform(X.toarray(), kmeans_labels)

# Create a helper function for saving the plots
def save_plot(fig, filename):
    fig.savefig(filename, format='png', dpi=300)
    print(f"Saved plot as {filename}")

# Plot PCA with K-means clusters and ellipses
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans_labels, cmap='viridis', edgecolors='k')
centers = kmeans.cluster_centers_

# Plot ellipses for each cluster
for i, center in enumerate(centers):
    cov = np.cov(pca_result[kmeans_labels == i].T)
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # Scale by 2 to visualize better
    angle = np.arctan2(w[1, 0], w[0, 0])
    angle = 180.0 * angle / np.pi  # Convert to degrees
    ell = Ellipse(xy=center, width=v[0], height=v[1], angle=angle, color='red', alpha=0.2)
    ax.add_patch(ell)

plt.legend(*scatter.legend_elements(), title="Clusters")
plt.title("PCA Plot with K-means Clusters and Ellipses")
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
save_plot(fig, "PCA_KMeans_Clusters.png")

# Plot LDA with 4 clusters
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(lda_result[:, 0], lda_result[:, 1], c=kmeans_labels, cmap='viridis', edgecolors='k')

plt.legend(*scatter.legend_elements(), title="Clusters")
plt.title("LDA Plot with 4 Clusters")
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
save_plot(fig, "LDA_Clusters.png")

# Additional clustering methods: DBSCAN and Agglomerative Clustering
# DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=4)
agg_labels = agg.fit_predict(X.toarray())

# Save clustering results to files
def save_clustering_results(labels, method_name):
    with open(f'clustering_{method_name}.txt', 'w') as f:
        for i in range(4):
            f.write(f"Cluster {i}:\n")
            cluster_abstracts = df['What is your talk abstract?'][labels == i].values
            for abstract in cluster_abstracts:
                f.write(f"Abstract: {abstract}\n")
                keywords = " ".join([word for word in vectorizer.get_feature_names_out() if word in abstract])
                f.write(f"Keywords: {keywords}\n\n")
            f.write("\n\n")

# Save K-means clusters
save_clustering_results(kmeans_labels, "kmeans")

# Save LDA clusters
save_clustering_results(kmeans_labels, "lda")  # LDA uses K-means labels

# Save DBSCAN clusters
save_clustering_results(dbscan_labels, "dbscan")

# Save Agglomerative clusters
save_clustering_results(agg_labels, "agg")
