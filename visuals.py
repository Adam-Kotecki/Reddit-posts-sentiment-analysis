import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk.tokenize import WhitespaceTokenizer

df = pd.read_excel('sentiment_analysis.xlsx')

# Count the occurrences of each sentiment label
sentiment_counts = df['sentiment'].value_counts()

# Define colors for each sentiment
colors = {
    'positive': 'lightgreen',
    'neutral': 'lightblue',
    'negative': 'lightcoral'
}

# Bar chart of sentiment count
plt.figure(figsize=(10, 8))
plt.bar(sentiment_counts.index, sentiment_counts.values, color=[colors[sent] for sent in sentiment_counts.index])
plt.xlabel('Sentiment', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.title('Distribution of Sentiment', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()
plt.savefig('assets/visual_1.png')

# Histogram of positive score
plt.figure(figsize=(10, 8))
plt.hist(df['title_positive'], bins=10, color='lightgreen', edgecolor='black')
plt.title('Positive Score Distribution', fontsize=18)
plt.xlabel('Positive score', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('assets/visual_2.png')

# Histogram of negative score
plt.figure(figsize=(10, 8))
plt.hist(df['title_negative'], bins=10, color='lightcoral', edgecolor='black')
plt.title('Negative Score Distribution', fontsize=18)
plt.xlabel('Negative score', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('assets/visual_3.png')

# Histogram of compound score
plt.figure(figsize=(10, 8))
plt.hist(df['title_overall'], bins=10, color='lightblue', edgecolor='black')
plt.title('Compound Score Distribution', fontsize=18)
plt.xlabel('Compound score', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('assets/visual_4.png')

# Histogram of votes
plt.figure(figsize=(10, 8))
plt.hist(df['votes'], bins=10, color='lightgrey', edgecolor='black')
plt.title('Votes Distribution', fontsize=18)
plt.xlabel('Votes', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('assets/visual_5.png')


def get_wordnet_pos(word):
    # [0] selects the first tuple in the returned list.
    # [1] selects the second element in the tuple, which is the POS tag.
    # [0] selects the first letter of the POS tag
    tag = nltk.pos_tag([word])[0][1][0].upper()
    # Mapper of POS symbol returned by pos_tag() to symbols regognizable for WordNetLemmatizer:
    tag_dict = {"J": 'a', "N": 'n', "V": 'v', "R": 'r'}
    # getting value based on key:
    return tag_dict.get(tag, 'n')


w_tokenizer = WhitespaceTokenizer()
lemmatizer = WordNetLemmatizer()

# Define the lemmatization function
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text.lower()) if w not in stopwords.words('english')]

# Apply the lemmatization function to the DataFrame
# to create new column with list of lemmas
df['lemmas'] = df['title'].apply(lemmatize_text)

# CountVectorizer() required input in form of string, not list
df['lemmas_str'] = df['lemmas'].apply(lambda x: ' '.join(x))

# Create a document-term matrix using CountVectorizer
vectorizer = CountVectorizer()
doc_term_matrix = vectorizer.fit_transform(df['lemmas_str'])

# Create gensim's Dictionary and Corpus from the original lemmas
# Create a dictionary representation of the documents
dictionary = corpora.Dictionary(df['lemmas'])

# Convert the document-term matrix into the format gensim expects
corpus = [dictionary.doc2bow(text) for text in df['lemmas']]

# Perform LDA using gensim
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, random_state=42, passes=10)

# Display the topics
topics = lda_model.print_topics(num_words=5)
for idx, topic in topics:
    print(f"Topic {idx + 1}: {topic}")




