#!/usr/bin/env python3
import pandas as pd

# Load your Excel file and sheet
df = pd.read_excel("Abstracts.xlsx", sheet_name="Talks")  # Adjust sheet name if needed

# Extract abstracts and drop any NaN values
df = df[['What is your first name?', 'What is your surname?', 'What is your talk abstract?']].dropna()

# List of keywords to search for
keywords = [
#    "speciation","divergence","diversification","hybrid","hybridization"
#     "model","models","theory","theories","software","mathematical"
      "Design","design"
]

# Function to check if any keyword is present in the text
def contains_keywords(text, keyword_list):
    """Check if a text contains any of the specified keywords."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in keyword_list)

# Filter abstracts containing any of the keywords
filtered_abstract = df[df['What is your talk abstract?'].apply(lambda x: contains_keywords(x, keywords))]

# Save the filtered abstracts to a text file
output_file = "filtered_abstracts.txt"

with open(output_file, 'w', encoding='utf-8') as f:
    for i, row in enumerate(filtered_abstract.itertuples(), start=1):
        first_name = row._1  # Index 1: 'What is your first name?'
        last_name = row._2   # Index 2: 'What is your surname?'
        abstract = row._3    # Index 3: 'What is your talk abstract?'
        
        f.write(f"Abstract {i} (Author: {first_name} {last_name}):\n{abstract}\n\n")
