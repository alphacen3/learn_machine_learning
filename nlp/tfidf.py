import math
from collections import Counter

def preprocess(text):
    stopwords = set(["the", "and", "is", "in", "it", "of", "to", "a"])  # Simplified example
    return [word.lower() for word in text.split() if word not in stopwords]

def calculate_tf(review):
    """Returns a dictionary of term frequencies for a review"""
    word_count = Counter(review)
    total_words = len(review)
    tf = {word: count / total_words for word, count in word_count.items()}
    return tf

def calculate_idf(reviews):
    """Returns a dictionary of inverse document frequencies for a list of reviews"""
    # number of doc
    # number of docs with each word in it
    df = Counter()
    for review in reviews:
        unique_words = set(review)
        df.update(unique_words)
    
    # note: avoid divide / 0
    return {word: math.log(len(reviews) / (df[word] + 1)) for word in df.keys()}

def calculate_tf_idf(reviews, mode="single"):
    # preprocess
    reviews = [preprocess(review) for review in reviews]

    # tf for each review
    tfs = [calculate_tf(review) for review in reviews]

    # idf
    idfs = calculate_idf(reviews)
    # tf-idf
    if mode == "single":
        tf_idfs = []
        for tf in tfs:
            tf_idf = {word: tf[word] * idfs[word] for word in tf.keys()}
            tf_idfs.append(tf_idf)
        return tf_idfs
    elif mode == "all":
        aggregate_tf_idf = Counter()
        for review, tf in zip(reviews, tfs):
            tf_idf = {word: tf[word] * idfs[word] for word in tf.keys()}
            aggregate_tf_idf.update(tf_idf)
        return aggregate_tf_idf

def get_top_k_keywords(tf_idfs, k, mode="single"):
    """Returns the top k keywords from a list of tf-idfs"""
    if mode == "single":
        top_keywords = []
        for tf_idf in tf_idfs:
            sorted_keywords = sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)
            top_keywords.append(sorted_keywords[:k])
        return top_keywords
    elif mode == "all":
        return [word for word, score in tf_idfs.most_common(k)]

# Example usage
reviews = ["This is the first review", "Here is another review", "A third one"]
tf_idfs = calculate_tf_idf(reviews, mode="single")
top_keywords = get_top_k_keywords(tf_idfs, 3, mode="single")
print(top_keywords)

tf_idfs = calculate_tf_idf(reviews, mode="all")
top_keywords = get_top_k_keywords(tf_idfs, 3, mode="all")
print(top_keywords)
