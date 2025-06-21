import argparse
import logging
import os
import time

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from tora import Tora

# Download required NLTK data
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("stopwords")


def safe_value(value):
    """
    Convert a value to a safe float or int, handling NaN, inf, bools, and strings.
    """
    if isinstance(value, (int, float)):
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, str):
        return None

    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def log_metric(client, name, value, step):
    """
    Log a metric to the Tora client if the value is valid.
    """
    value = safe_value(value)
    if value is not None:
        client.log(name=name, value=value, step=step)


def preprocess_text(text, remove_stopwords=True, lemmatize=True):
    """
    Tokenize, lowercase, remove stopwords, and optionally lemmatize text.
    """
    tokens = word_tokenize(text.lower())

    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in stop_words and len(t) > 3]

    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def process_documents(docs, remove_stopwords=True, lemmatize=True):
    """
    Apply preprocessing to a list of documents.
    """
    return [preprocess_text(d, remove_stopwords, lemmatize) for d in docs]


def extract_topics(model, vectorizer, num_words=10):
    """
    Extract top words for each topic from an LDA model.
    """
    feature_names = vectorizer.get_feature_names_out()
    topics = {}

    for idx, topic in enumerate(model.components_):
        top_idx = topic.argsort()[: -num_words - 1 : -1]
        topics[f"topic_{idx}"] = [(feature_names[i], topic[i]) for i in top_idx]

    return topics


def compute_perplexity(model, X):
    """
    Compute the perplexity of the LDA model on the data X.
    """
    return model.perplexity(X)


def compute_topic_diversity(model, vectorizer, num_words=50):
    """
    Calculate average pairwise topic diversity based on top words.
    """
    feature_names = vectorizer.get_feature_names_out()
    term_sets = []

    for topic in model.components_:
        top_idx = topic.argsort()[: -num_words - 1 : -1]
        term_sets.append(set(feature_names[i] for i in top_idx))

    diversity, count = 0.0, 0
    for i in range(len(term_sets)):
        for j in range(i + 1, len(term_sets)):
            inter = term_sets[i] & term_sets[j]
            uni = term_sets[i] | term_sets[j]
            diversity += 1 - len(inter) / len(uni)
            count += 1

    return diversity / count if count else 0


def evaluate_model(model, X, vectorizer, tora_client, step):
    """
    Evaluate and log perplexity and topic diversity, printing top terms.
    """
    perp = compute_perplexity(model, X)
    log_metric(tora_client, "perplexity", perp, step)
    print(f"Perplexity: {perp:.4f}")

    div = compute_topic_diversity(model, vectorizer)
    log_metric(tora_client, "topic_diversity", div, step)
    print(f"Topic Diversity: {div:.4f}")

    topics = extract_topics(model, vectorizer)
    for tid, terms in topics.items():
        terms_str = ", ".join(t for t, _ in terms)
        print(f"{tid}: {terms_str}")

    return perp, div, topics


def load_dataset_data(name, field=None, max_samples=None):
    """
    Load text data from a specified dataset by name.
    Supports 20newsgroups, IMDb, AG News, or other Hugging Face datasets.
    """
    name_lower = name.lower()

    if name_lower in ["20newsgroups", "20_newsgroups"]:
        data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
        texts = data.data

    elif name_lower == "imdb":
        from datasets import load_dataset

        ds = load_dataset("imdb")
        texts = ds["train"]["text"]

    elif name_lower == "ag_news":
        from datasets import load_dataset

        ds = load_dataset("ag_news")
        texts = ds["train"]["text"]

    else:
        from datasets import load_dataset

        ds = load_dataset(name)
        if field is None:
            for f in ["text", "content", "article", "body"]:
                if f in ds["train"].features:
                    field = f
                    break
            if field is None:
                raise ValueError(f"No text field in dataset {name}")

        split = "train" if "train" in ds else "test"
        texts = ds[split][field]

    if max_samples and max_samples < len(texts):
        texts = texts[:max_samples]

    return texts


def train_lda_model(args):
    """
    Train LDA models over a range of topic counts, logging and saving the best.
    """
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )

    tora = Tora.create_experiment(
        name=f"LDA_{args.dataset}",
        description="Topic modeling with sklearn LDA",
        hyperparams=vars(args),
    )
    start_time = time.time()

    texts = load_dataset_data(args.dataset, args.text_field, args.max_samples)
    print(f"Loaded {len(texts)} documents")

    processed = process_documents(texts, args.remove_stopwords, args.lemmatize)
    vectorizer = CountVectorizer(
        max_df=args.max_doc_freq,
        min_df=args.min_doc_freq,
        max_features=args.keep_n,
    )
    X = vectorizer.fit_transform(processed)

    best_model = None
    best_perp = float("inf")

    for k in range(args.start, args.end + 1, args.step):
        print(f"Training LDA with {k} topics...")
        lda = LatentDirichletAllocation(
            n_components=k,
            max_iter=args.iterations,
            learning_method=args.learning_method,
            random_state=args.seed,
            n_jobs=-1,
        )
        lda.fit(X)

        perp, _, _ = evaluate_model(lda, X, vectorizer, tora, k)
        if perp < best_perp:
            best_perp = perp
            best_model = lda

    print(f"Best model: {best_model.n_components} topics, perplexity={best_perp:.4f}")

    if args.save_model:
        import joblib

        os.makedirs("results", exist_ok=True)
        path = f"results/lda_{args.dataset}_{best_model.n_components}.joblib"
        joblib.dump(best_model, path)
        print(f"Model saved to {path}")

    duration = time.time() - start_time
    log_metric(tora, "execution_time", duration, args.end)
    print(f"Total execution time: {duration:.2f}s")

    tora.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LDA topic models on text datasets"
    )
    parser.add_argument("--dataset", default="20newsgroups", help="Name of the dataset")
    parser.add_argument(
        "--text_field", default=None, help="Field name for text in datasets"
    )
    parser.add_argument(
        "--start", type=int, default=5, help="Starting number of topics"
    )
    parser.add_argument("--end", type=int, default=20, help="Ending number of topics")
    parser.add_argument("--step", type=int, default=5, help="Step size for topic range")
    parser.add_argument(
        "--iterations", type=int, default=100, help="Max LDA iterations"
    )
    parser.add_argument(
        "--learning_method",
        choices=["batch", "online"],
        default="batch",
        help="Learning method for LDA",
    )
    parser.add_argument(
        "--min_doc_freq",
        type=int,
        default=5,
        help="Minimum document frequency for vocabulary",
    )
    parser.add_argument(
        "--max_doc_freq",
        type=float,
        default=0.5,
        help="Maximum document frequency (proportion)",
    )
    parser.add_argument(
        "--keep_n", type=int, default=100000, help="Maximum vocabulary size"
    )
    parser.add_argument(
        "--remove_stopwords",
        action="store_true",
        default=True,
        help="Remove English stopwords",
    )
    parser.add_argument(
        "--lemmatize", action="store_true", default=True, help="Apply lemmatization"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to load",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save_model",
        action="store_true",
        default=False,
        help="Whether to save the best model",
    )

    args = parser.parse_args()
    train_lda_model(args)
