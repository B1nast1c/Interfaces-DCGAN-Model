import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from utils import common

# nltk.download('wordnet')
# nltk.download('punkt')


def save_model(sentences):
    model = Word2Vec(sentences, vector_size=300, window=3,
                     min_count=1, sg=0, workers=4)
    model.save(common.WORD_MODEL_LOCATION + '/word2vec.model')


def create_model():
    words = ['a', 'webpage', 'interface']
    numeric_words = [word_tokenize(word) for word in words]
    tokenized_words = numeric_words

    for single_class in range(len(common.KEYWORDS)):
        tokens = common.KEYWORDS[single_class]
        for token in tokens:
            numeric_token = word_tokenize(token)
            tokenized_words.append(numeric_token)

    save_model(tokenized_words)


def load_model():
    model = Word2Vec.load(common.WORD_MODEL_LOCATION + '/word2vec.model')
    word_vectors = model.wv
    word_to_vec = {word: word_vectors[word]
                   for word in word_vectors.index_to_key}
    return word_to_vec
