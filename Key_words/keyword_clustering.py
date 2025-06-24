import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# Load your DataFrame
df = pd.read_excel("Abstracts.xlsx", sheet_name="Talks")  # Adjust as needed

# Extract abstracts and drop any NaN values
abstracts = df['What is your talk abstract?'].dropna()
print(abstracts.size)

# List of 20 predefined keywords (this can be adjusted)
keywords = [
    "hybridization", "speciation", "balancing", "modelling", "duplication", "ploidy", "intraspecific", "quantitative","bioinformatic","tRNA","mRNA","whole genome","multi-locus",
    "life-history", "longevity", "demography", "convergence", "selection", "transposable", "sex","DFE","pipeline","multilocus","locus",
    "inbreeding", "haplotype", "drift", "coloration", "conservation", "impute", "imputation", "sex-chromosome", "colour","genome evolution","viral","retrovirus",
    "aDNA", "models", "effective population size", "natural selection", "sexual chromosome", "purifying","theory","transloc","infection","tree","parameter","transplant",
    "purifying selection", "phylogeography", "algorithm", "model", "NGS", "adaptative", "human", "radiation","virus","pest","urban","Darwin","coalescent",
    "bottleneck", "transfer", "heritability", "evolvability", "barrier", "admixture", "demographic history", "RNA","bacteria","sequencing","parental","ecotype",
    "transcriptome", "SVs", "statistics", "genomics", "WGS", "low coverage", "color", "mimicry", "coverage","fungi","life history","plot","evolutionary origin","sympatric",
    "recombination", "recombination rate", "reproductive", "mating", "phylogenetic", "phylogeny", "mutation", "X-linked","extinction","genetic basis","molecular basis","biodiversity",
    "y-linked", "ARG", "Ne", "SFS", "tool", "method", "simulations", "development", "machine learning", "mutations","animal","microbiome","adapted","copy number",
    "fitness", "meiotic", "differentiation", "divergence", "heterozygosity", "inversions", "kinship", "regulator","plant","endosymbiont","gene loss","reference genome",
    "regulation", "sympatry", "hybridising", "convergent", "conflict", "male", "female", "simulated", "genomic architecture","organel","variance","pedigree","inference",
    "load", "deleterious", "small population", "genetic drift", "linked", "linkage", "adaptation", "population structure","chloropolast","decline","genetic architecture","plastic",
    "disease", "Inbreeding depression", "Barriers", "gene flow", "transposon", "structural variant","theoretical","mitochondrial","mitocondria","complex trait","lineage",
    "sampling", "fragmentation","evolutionary history","supergene","supergenes","mate","introgression","phylogenomics","adaptive","lc-WGS","gene expression","selective pressure"
]

#
#keywords = [
#    "speciation","divergence","diversification","hybrid","hybridization","model","models","theory","theories","software","mathematical"
#]

# Create a binary vectorizer based on the keywords
vectorizer = CountVectorizer(vocabulary=keywords, binary=True)

# Transform the abstracts into binary feature vectors
X_binary = vectorizer.transform(abstracts)

# Apply K-means clustering on the binary vectors
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X_binary)

# Plot the clustering result in a 2D space (PCA for dimensionality reduction)
from sklearn.decomposition import PCA

# Reduce to 2 components for plotting
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_binary.toarray())  # Converting sparse matrix to dense for PCA
print(len(pca_result))

# Plot the clustering results
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans_labels, cmap='rainbow', edgecolors='k')

plt.legend(*scatter.legend_elements(), title="Clusters")
plt.title("K-means Clustering Based on Keywords (Presence/Absence)")
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

# Save the plot
fig.savefig("Keyword_Based_KMeans_Clusters.png", format='png', dpi=300)
plt.show()

# Save clustering results with top keywords
def save_clustering_results(labels, abstracts, vectorizer):
    feature_names = vectorizer.get_feature_names_out()  # Keywords from the CountVectorizer

    with open('keyword_based_clustering_results.txt', 'w') as f:
        for label in np.unique(labels):  # Iterate over unique cluster labels
            f.write(f"Cluster {label}:\n\n")
            cluster_indices = np.where(labels == label)[0]
            
            # Write abstracts in the current cluster
            for idx in cluster_indices:
                f.write(f"Abstract {idx + 1}:\n{abstracts.iloc[idx]}\n\n")
            
            # Write the keywords present in the cluster (just a binary count)
            cluster_keywords = X_binary[cluster_indices].sum(axis=0).A1  # Sum across abstracts for each keyword
            top_keywords_indices = np.argsort(cluster_keywords)[::-1][:10]
            top_keywords = [feature_names[i] for i in top_keywords_indices]
            
            # Write top words (keywords that best define the cluster)
            f.write(f"Top words:\n{', '.join(top_keywords)}\n\n\n")

# Save K-means clusters with top keywords for the keyword-based clustering
save_clustering_results(kmeans_labels, abstracts, vectorizer)
