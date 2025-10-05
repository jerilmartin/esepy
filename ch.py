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



# Install if not done:
# pip install gensim matplotlib nltk

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec
from gensim.downloader import load
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

nltk.download('punkt')

# --- Text Input ---
text = """
I love studying NLP. NLP is fun and very useful.
I study NLP every day to improve my skills.
Deep learning is related to NLP and AI.
"""

# --- Tokenize sentences and words ---
sentences = [word_tokenize(s.lower()) for s in sent_tokenize(text)]

# --- 1️⃣ Train Word2Vec (on our text) ---
w2v = Word2Vec(sentences, vector_size=50, window=3, min_count=1)

print("\n🔹 Word2Vec Similar words to 'nlp':")
print(w2v.wv.most_similar('nlp'))

# --- 2️⃣ Load Pretrained GloVe ---
glove = load('glove-wiki-gigaword-50')

print("\n🔹 GloVe Similar words to 'learning':")
print(glove.most_similar('learning'))

# --- 3️⃣ Visualize Word2Vec ---
pca = PCA(n_components=2)
X = pca.fit_transform(w2v.wv.vectors)

plt.figure(figsize=(6,4))
plt.scatter(X[:, 0], X[:, 1])
for i, word in enumerate(w2v.wv.index_to_key):
    plt.text(X[i, 0], X[i, 1], word)
plt.title("Word2Vec (Trained on Our Text)")
plt.show()

# --- 4️⃣ Visualize few GloVe words ---
words = ['ai', 'learning', 'data', 'model', 'computer']
vectors = [glove[w] for w in words]
X = PCA(n_components=2).fit_transform(vectors)

plt.figure(figsize=(6,4))
plt.scatter(X[:, 0], X[:, 1])
for i, word in enumerate(words):
    plt.text(X[i, 0], X[i, 1], word)
plt.title("GloVe (Pretrained) Embeddings")
plt.show()





from nltk.wsd import lesk
sentence = "I went to the bank to deposit some money"

# Tokenize sentence
tokens = word_tokenize(sentence)

# Apply Lesk algorithm
ambiguous_word = "bank"
sense = lesk(tokens, ambiguous_word)

# Display results
print(f"Sentence: {sentence}")
print(f"Word: {ambiguous_word}")
print(f"Predicted Sense: {sense}")
print(f"Definition: {sense.definition()}")



