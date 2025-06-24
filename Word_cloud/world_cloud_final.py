import pandas as pd
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk import pos_tag

# Download the stopwords from NLTK 
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

# Load your DataFrame 
excel_path = "Abstracts.xlsx" 
xls = pd.ExcelFile(excel_path)

# Extract abstracts from all sheets
all_abstracts = []

for sheet_name in xls.sheet_names:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    if 'What is your talk abstract?' in df.columns:
        abstracts = df['What is your talk abstract?'].dropna().tolist()
        all_abstracts.extend(abstracts)

# Convert to a single pandas Series
abstracts = pd.Series(all_abstracts)

# Create a list of custom unwanted words
unwanted_words = ["previous", "highly", "identified", "potential", "high", "increased", "using", "the", "and","type","types","approaches","approach","analyses","analysis","differences","framework","context","project","effect",
"of","work","panel","little","including","used","analysed","study","studies","remains","key","understanding","new","talk","presence","level","insight","populations","genes","insight","extent","result","results","impact",
"effects","years","measures","lines","research","consequences","times","number","hundreds","hundred"]

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
    exceptions = ["locus","process","across","fitness"]

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

# Sum the TF-IDF values for each word across all abstracts
word_freq = X.sum(axis=0).A1

# Get the corresponding words (terms)
words = vectorizer.get_feature_names_out()

# Create a dictionary of word frequencies
word_freq_dict = dict(zip(words, word_freq))

# Print the list of unique words and their frequencies
print("Unique words used in the word cloud:")
for word, freq in word_freq_dict.items():
    print(f"{word}: {freq}")

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq_dict)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Save the word cloud image
wordcloud.to_file("wordcloud.png")
