import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize, TreebankWordTokenizer,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from nltk.corpus import wordnet




nltk.download('stopwords')
nltk.download('wordnet')

#installs:  pip install textblob
#python -m textblob.download_corpora

#pip install spacy
#python -m spacy download en_core_web_sm

#pip install gensim

text_non_english = "¡Hola! Los gatos corren más rápido que los perros."
text_non_english = text_non_english.lower()
tokens_non_english = word_tokenize(text_non_english, language='spanish')
print(" Tokens (Spanish):", tokens_non_english)



text = "Hello!!! Cats, dogs, and birds are running faster than ever in 2025."
text = text.lower()
# 2. Tokenization (method 1): Basic NLTK tokenizer
tokens_basic = word_tokenize(text)
treebank_tokenizer = TreebankWordTokenizer()
tokens_treebank = treebank_tokenizer.tokenize(text)

stop_words = set(stopwords.words('english'))

filtered_basic = [w for w in tokens_basic if w.isalpha() and w not in stop_words]
filtered_treebank = [w for w in tokens_treebank if w.isalpha() and w not in stop_words]

print(" word_tokenize:", tokens_basic)
print("Filtered (word_tokenize):", filtered_basic, "\n")

print(" TreebankWordTokenizer:", tokens_treebank)
print("Filtered (Treebank):", filtered_treebank, "\n")

filtfreq = Counter(filtered_treebank)
print(filtfreq)


ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed_basic = [ps.stem(w) for w in filtered_basic]
lemmatized_basic = [lemmatizer.lemmatize(w) for w in filtered_basic]
print("After Stemming (Porter):", stemmed_basic)
print("After Lemmatization:", lemmatized_basic)



# Word to analyze
word = "happy"

# Get all synsets (word senses)
synsets = wordnet.synsets(word)
print(f"Synsets of '{word}':", synsets)

# --- Extract Synonyms ---
synonyms = set()
for syn in synsets:
    for lemma in syn.lemmas():
        synonyms.add(lemma.name())

# --- Extract Antonyms ---
antonyms = set()
for syn in synsets:
    for lemma in syn.lemmas():
        if lemma.antonyms():
            antonyms.add(lemma.antonyms()[0].name())

# --- Extract Hypernyms (General Categories) ---
hypernyms = set()
for syn in synsets:
    for hyper in syn.hypernyms():
        hypernyms.add(hyper.name().split('.')[0])

print(f"\n Synonyms of '{word}':", synonyms)
print(f" Antonyms of '{word}':", antonyms)
print(f" Hypernyms of '{word}':", hypernyms)


text = "I want to increase my knowledge and to increase my skills"

tokens = nltk.word_tokenize(text.lower())
bigram_list = list(nltk.bigrams(tokens))

# --- Count unigrams and bigrams ---
unigram_counts = Counter(tokens)
bigram_counts = Counter(bigram_list)

# Vocabulary size
vocab_size = len(set(tokens))

# Choose bigram to calculate probability
w1, w2 = "to", "increase"

# --- Without Smoothing ---
if unigram_counts[w1] > 0:
    prob_no_smooth = bigram_counts[(w1, w2)] / unigram_counts[w1]
else:
    prob_no_smooth = 0

# --- With Laplace Smoothing ---
prob_laplace = (bigram_counts[(w1, w2)] + 1) / (unigram_counts[w1] + vocab_size)

# --- Print results ---
print(f"P('{w2}' | '{w1}') without smoothing = {prob_no_smooth:.4f}")
print(f"P('{w2}' | '{w1}') with Laplace smoothing = {prob_laplace:.4f}")


import spacy

# Load English model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "Apple Inc. was founded by Steve Jobs in California on April 1, 1976."

# Process text
doc = nlp(text)

# Extract named entities
print("Named Entities:")
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")
from textblob import TextBlob

# Sample sentences
sentences = [
    "I love studying NLP, it's amazing!",
    "I hate when my code doesn't run.",
    "The weather is okay today."
]

# Analyze sentiment
for sent in sentences:
    blob = TextBlob(sent)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    print(f"'{sent}' → {sentiment} (Polarity: {polarity})")



from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA




text = """
I love studying NLP. NLP is fun and very useful. 
I study NLP every day to improve my skills. 
Deep learning is related to NLP and AI.
"""


sentences = sent_tokenize(text)


tokenized_sentences = [word_tokenize(sent.lower()) for sent in sentences]


model = Word2Vec(tokenized_sentences, vector_size=50, window=2, min_count=1, workers=1)


words = list(model.wv.index_to_key)
vectors = model.wv[words]

vectors_2d = PCA(n_components=2).fit_transform(vectors)

plt.figure(figsize=(6,4))
plt.scatter(vectors_2d[:,0], vectors_2d[:,1])
for i, word in enumerate(words):
    plt.text(vectors_2d[i,0]+0.01, vectors_2d[i,1]+0.01, word)
plt.title("Word2Vec 2D Visualization")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
