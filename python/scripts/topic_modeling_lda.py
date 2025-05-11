import argparse


import logging


import time


import os


import numpy as np


import nltk


from nltk.stem import WordNetLemmatizer


from nltk.tokenize import word_tokenize


from nltk.corpus import stopwords


from sklearn.feature_extraction.text import CountVectorizer


from sklearn.decomposition import LatentDirichletAllocation


from sklearn.datasets import fetch_20newsgroups


from tora import Tora


nltk.download("punkt")


nltk.download("punkt_tab")


nltk.download("wordnet")


nltk.download("stopwords")


def safe_value(value):
    if isinstance(value, (int, float)):
        if np.isnan(value) or np.isinf(value):
            return 0.0

        return float(value)

    elif isinstance(value, bool):
        return int(value)

    elif isinstance(value, str):
        return None

    else:
        try:
            return float(value)

        except (ValueError, TypeError):
            return None


def log_metric(client, name, value, step):
    value = safe_value(value)

    if value is not None:
        client.log(name=name, value=value, step=step)


def preprocess_text(text, remove_stopwords=True, lemmatize=True):
    tokens = word_tokenize(text.lower())

    if remove_stopwords:
        stop_words = set(stopwords.words("english"))

        tokens = [t for t in tokens if t not in stop_words and len(t) > 3]

    if lemmatize:
        lemmatizer = WordNetLemmatizer()

        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def process_documents(docs, remove_stopwords=True, lemmatize=True):
    return [preprocess_text(doc, remove_stopwords, lemmatize) for doc in docs]


def extract_topics(model, vectorizer, num_words=10):
    feature_names = vectorizer.get_feature_names_out()

    topics = {}

    for idx, topic in enumerate(model.components_):
        top_idx = topic.argsort()[: -num_words - 1 : -1]

        topics[f"topic_{idx}"] = [(feature_names[i], topic[i]) for i in top_idx]

    return topics


def compute_perplexity(model, X):
    return model.perplexity(X)


def compute_topic_diversity(model, vectorizer, num_words=50):
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
    perp = compute_perplexity(model, X)

    log_metric(tora_client, "perplexity", perp, step)

    print(f"Perplexity: {perp:.4f}")

    div = compute_topic_diversity(model, vectorizer)

    log_metric(tora_client, "topic_diversity", div, step)

    print(f"Topic Diversity: {div:.4f}")

    topics = extract_topics(model, vectorizer)

    for tid, terms in topics.items():
        print(f"{tid}: " + ", ".join([t for t, _ in terms]))

    return perp, div, topics


def load_dataset_data(name, field=None, max_samples=None):
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

        texts = ds["train"][field] if "train" in ds else ds["test"][field]

    if max_samples and max_samples < len(texts):
        texts = texts[:max_samples]

    return texts


def train_lda_model(args):
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

    vec = CountVectorizer(
        max_df=args.max_doc_freq, min_df=args.min_doc_freq, max_features=args.keep_n
    )

    X = vec.fit_transform(processed)

    best_model, best_perp = None, float("inf")

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

        perp, _, _ = evaluate_model(lda, X, vec, tora, k)

        if perp < best_perp:
            best_perp, best_model = perp, lda

    print(f"Best model: {best_model.n_components} topics, perplexity={best_perp:.4f}")

    if args.save_model:
        import joblib

        os.makedirs("results", exist_ok=True)

        joblib.dump(
            best_model, f"results/lda_{args.dataset}_{best_model.n_components}.joblib"
        )

        print("Model saved to results/")

    duration = time.time() - start_time

    log_metric(tora, "execution_time", duration, args.end)

    print(f"Total execution time: {duration:.2f}s")

    tora.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="20newsgroups")

    parser.add_argument("--text_field", default=None)

    parser.add_argument("--start", type=int, default=5)

    parser.add_argument("--end", type=int, default=20)

    parser.add_argument("--step", type=int, default=5)

    parser.add_argument("--iterations", type=int, default=100)

    parser.add_argument(
        "--learning_method", choices=["batch", "online"], default="batch"
    )

    parser.add_argument("--min_doc_freq", type=int, default=5)

    parser.add_argument("--max_doc_freq", type=float, default=0.5)

    parser.add_argument("--keep_n", type=int, default=100000)

    parser.add_argument("--remove_stopwords", action="store_true", default=True)

    parser.add_argument("--lemmatize", action="store_true", default=True)

    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--save_model", action="store_true", default=False)

    args = parser.parse_args()

    train_lda_model(args)
