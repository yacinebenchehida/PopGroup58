import pandas as pd
import re

# Define your list of keywords
keywords = [
    "hybridization", "speciation", "balancing", "modelling", "duplication", "ploidy", "intraspecific", "quantitative","indel",
    "bioinformatic", "tRNA", "mRNA", "whole genome", "multi-locus", "life-history", "longevity", "demography","synonymous",
    "convergence", "selection", "transposable", "sex", "DFE", "pipeline", "multilocus", "locus", "inbreeding","sweep",
    "haplotype", "drift", "coloration", "conservation", "impute", "imputation", "sex-chromosome", "colour","positive selection",
    "genome evolution", "viral", "retrovirus", "aDNA", "models", "effective population size", "natural selection",
    "sexual chromosome", "purifying", "theory", "transloc", "infection", "tree", "parameter", "transplant",
    "purifying selection", "phylogeography", "algorithm", "model", "NGS", "adaptative", "human", "radiation",
    "virus", "pest", "urban", "Darwin", "coalescent", "bottleneck", "transfer", "heritability", "evolvability",
    "barrier", "admixture", "demographic history", "RNA", "bacteria", "sequencing", "parental", "ecotype","mito-nuclear",
    "transcriptome", "SVs", "statistics", "genomics", "WGS", "low coverage", "color", "mimicry", "coverage","phylogenies","neutral",
    "fungi", "life history", "plot", "evolutionary origin", "sympatric", "recombination", "recombination rate",
    "reproductive", "mating", "phylogenetic", "phylogeny", "mutation", "X-linked", "extinction", "genetic basis",
    "molecular basis", "biodiversity", "y-linked", "ARG", "Ne", "SFS", "tool", "method", "simulations", "development",
    "machine learning", "mutations", "animal", "microbiome", "adapted", "copy number", "fitness", "meiotic",
    "differentiation", "divergence", "heterozygosity", "inversions", "kinship", "regulator", "plant", "endosymbiont",
    "gene loss", "reference genome", "regulation", "sympatry", "hybridising", "convergent", "conflict", "male",
    "female", "simulated", "genomic architecture", "organel", "variance", "pedigree", "inference", "load",
    "deleterious", "small population", "genetic drift", "linked", "linkage", "adaptation", "population structure",
    "chloropolast", "decline", "genetic architecture", "plastic", "disease", "Inbreeding depression", "Barriers",
    "gene flow", "transposon", "structural", "theoretical", "mitochondrial", "mitocondria", "complex trait","deletion",
    "lineage", "sampling", "fragmentation", "evolutionary history", "supergene", "supergenes", "mate","insertion",
    "introgression", "phylogenomics", "adaptive", "lc-WGS", "gene expression", "selective pressure","hitchhiking","method"
]

# Function to process abstracts and search for whole keywords
def process_abstracts(abstracts, keywords):
    results = []
    
    for idx, abstract in enumerate(abstracts, start=1):  # Keep track of abstract number
        abstract = abstract.replace(",", "")
        found_keywords = set()  # Store unique keywords
        for keyword in keywords:
            # Match keyword as a whole word using \b
            pattern = r'\b' + re.escape(keyword)
            if re.search(pattern, abstract, re.IGNORECASE):
                found_keywords.add(keyword)
        
        # Append abstract number, abstract, count, and keywords found
        results.append([
            idx,  # Abstract number
            abstract,
            len(found_keywords),
            ' '.join(found_keywords)
        ])
    
    # Return as DataFrame with strict column names
    return pd.DataFrame(results, columns=[
        'Abstract Number',
        'What is your talk abstract?',
        'Number of keywords',
        'List of keywords'
    ])

# Load the input Excel file (sheet "Talks")
df = pd.read_excel("Abstracts.xlsx", sheet_name="Talks")  # Adjust file path if needed
abstracts = df['What is your talk abstract?'].dropna()

# Process the abstracts
output_df = process_abstracts(abstracts, keywords)

# Add Full Name column (combine "What is your first name?" and "What is your surname?")
df_clean = df.dropna(subset=['What is your first name?', 'What is your surname?','what is your talk title?'])
output_df['Full Name'] = df_clean['What is your first name?'].str.strip() + ' ' + df_clean['What is your surname?'].str.strip()
output_df['Title'] = df_clean['what is your talk title?']

# Save the output DataFrame to a new CSV file
output_df.to_csv("Abstracts_with_keywords.csv", index=False)
print("Output saved to 'Abstracts_with_keywords.csv'")
