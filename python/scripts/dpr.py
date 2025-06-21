import os
import random
from collections import defaultdict

import numpy as np  # Import numpy to handle potential numpy types
import torch
from datasets import Dataset, load_dataset
from sentence_transformers import InputExample, SentenceTransformer, losses, util
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

from tora import Tora  # Assuming tora_client.py is in the same directory

# --- Configuration ---
MODEL_NAME = (
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"  # A good starting point for DPR
)
BATCH_SIZE = 16
NUM_EPOCHS = 25
LEARNING_RATE = 2e-5
WARMUP_STEPS = 100
DATASET_NAME = "nq_open"
TRAIN_SPLIT = "train"
EVAL_SPLIT = "validation"
MAX_TRAIN_SAMPLES = 5000  # Limit for demonstration purposes
MAX_EVAL_SAMPLES = 500  # Limit for demonstration purposes
OUTPUT_MODEL_PATH = "results/dpr_model_hf_dataset"
TORA_EXPERIMENT_NAME = "Dense Passage Retrieval Training (NQ-Open)"
TORA_DESCRIPTION = (
    "Training a Dense Passage Retrieval model on NQ-Open dataset with Tora logging"
)
TORA_TAGS = ["DPR", "SentenceTransformers", "Information Retrieval", "NQ-Open", "ML"]


# --- Tora-integrated Information Retrieval Evaluator ---
class ToraInformationRetrievalEvaluator(InformationRetrievalEvaluator):
    """
    A custom InformationRetrievalEvaluator that logs evaluation metrics to Tora.
    This class extends the standard SentenceTransformers InformationRetrievalEvaluator
    and overrides its __call__ method to include Tora logging.
    """

    def __init__(self, tora_client: Tora, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tora_client = tora_client
        self.global_step = 0  # To track training steps for logging

    def __call__(
        self, model, output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> float:
        """
        This method is called periodically during training to evaluate the model.
        It computes standard IR metrics and logs them to Tora.

        Args:
            model: The SentenceTransformer model being evaluated.
            output_path (str): Path to save evaluation results (optional).
            epoch (int): Current training epoch.
            steps (int): Current global training step.

        Returns:
            float: The main evaluation score (e.g., MRR@10).
        """
        metrics_dict = super().__call__(model, output_path, epoch, steps)

        self.global_step = steps if steps != -1 else self.global_step

        # Log each metric to Tora
        for metric_name, metric_value in metrics_dict.items():
            try:
                value_to_log = float(metric_value)

                # Handle NaN or Inf values for Tora compatibility
                if np.isnan(value_to_log):
                    # print(f"Warning: Encountered NaN value for '{metric_name}' at step {self.global_step}. Logging as 0.0.") # Debug print
                    value_to_log = 0.0
                elif np.isinf(value_to_log):
                    # print(f"Warning: Encountered Inf value for '{metric_name}' at step {self.global_step}. Logging as a large number.") # Debug print
                    value_to_log = 1e9

                self.tora_client.log(
                    name=metric_name, value=value_to_log, step=self.global_step
                )
                # print(f"Successfully logged {metric_name}: {value_to_log} at step {self.global_step}") # Debug print
            except Exception as e:
                # print(f"Error logging '{metric_name}' to Tora: {e}") # Debug print
                # print(f"Attempted to log: name='{metric_name}', value={metric_value} (type: {type(metric_value)}), step={self.global_step}") # Debug print
                pass  # Suppress error prints for non-critical logging failures

        main_score = metrics_dict.get("dpr_evaluation_cosine_mrr@10", 0.0)
        return main_score


# --- Main Training Script ---
if __name__ == "__main__":
    # 1. Initialize Tora Experiment
    tora = Tora.create_experiment(
        name=TORA_EXPERIMENT_NAME,
        description=TORA_DESCRIPTION,
        hyperparams={
            "model_name": MODEL_NAME,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "warmup_steps": WARMUP_STEPS,
            "loss_function": "MultipleNegativesRankingLoss",
            "dataset": DATASET_NAME,
            "train_samples": MAX_TRAIN_SAMPLES,
            "eval_samples": MAX_EVAL_SAMPLES,
        },
        tags=TORA_TAGS,
    )
    print(
        f"Tora Experiment created with ID: {tora._experiment_id}"
    )  # Keep this one for user to know experiment ID

    # 2. Load Pre-trained Sentence Transformer Model
    model = SentenceTransformer(MODEL_NAME)

    # 3. Load Hugging Face Dataset
    train_hf_dataset = (
        load_dataset(DATASET_NAME, split=TRAIN_SPLIT)
        .shuffle(seed=42)
        .select(range(MAX_TRAIN_SAMPLES))
    )
    eval_hf_dataset = (
        load_dataset(DATASET_NAME, split=EVAL_SPLIT)
        .shuffle(seed=42)
        .select(range(MAX_EVAL_SAMPLES))
    )

    # 4. Prepare Training Data for InputExample
    train_data = []
    for example in train_hf_dataset:
        question = example["question"]
        if example["answer"] and len(example["answer"]) > 0:
            positive_passage = example["answer"][0]
            train_data.append(InputExample(texts=[question, positive_passage]))
        else:
            pass  # Skip examples without answers

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)

    # 5. Define Training Loss Function
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # 6. Prepare Evaluation Data for InformationRetrievalEvaluator
    eval_queries = {}
    eval_corpus = {}
    eval_relevant_docs = defaultdict(set)

    query_id_counter = 0
    passage_id_counter = 0

    for example in eval_hf_dataset:
        question = example["question"]
        if example["answer"] and len(example["answer"]) > 0:
            q_id = f"q_{query_id_counter}"
            eval_queries[q_id] = question
            query_id_counter += 1

            p_id = f"p_{passage_id_counter}"
            eval_corpus[p_id] = example["answer"][0]
            passage_id_counter += 1

            eval_relevant_docs[q_id].add(p_id)

    # Add some additional passages to the corpus that are not direct answers
    # to make the retrieval task slightly more challenging for this small synthetic set.
    distractor_passages = [
        "The Eiffel Tower is a famous landmark in Paris.",
        "Artificial intelligence is a rapidly growing field.",
        "Plants use chlorophyll to absorb sunlight.",
        "Shakespeare's plays are often studied in literature classes.",
        "Elephants are the largest land animals.",
    ]
    for dp in distractor_passages:
        p_id = f"p_{passage_id_counter}"
        eval_corpus[p_id] = dp
        passage_id_counter += 1

    # Print some samples of queries and corpus
    print("\n--- Sample Queries from Evaluation Set ---")
    for i, (q_id, query_text) in enumerate(eval_queries.items()):
        if i >= 3:  # Print first 3 queries
            break
        print(f"Query ID: {q_id}, Text: '{query_text}'")

    print("\n--- Sample Passages from Evaluation Corpus ---")
    for i, (p_id, passage_text) in enumerate(eval_corpus.items()):
        if i >= 3:  # Print first 3 passages
            break
        print(f"Passage ID: {p_id}, Text: '{passage_text}'")
    print("------------------------------------------\n")

    ir_evaluator = ToraInformationRetrievalEvaluator(
        tora_client=tora,
        queries=eval_queries,
        corpus=eval_corpus,
        relevant_docs=eval_relevant_docs,
        name="dpr_evaluation",
        show_progress_bar=True,
        batch_size=BATCH_SIZE,
    )

    # 7. Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=ir_evaluator,
        epochs=NUM_EPOCHS,
        warmup_steps=WARMUP_STEPS,
        output_path=OUTPUT_MODEL_PATH,
        checkpoint_path=OUTPUT_MODEL_PATH,
        checkpoint_save_steps=100,
        show_progress_bar=True,
    )

    # 8. Save the final fine-tuned model
    final_model_save_path = os.path.join(OUTPUT_MODEL_PATH, "final_model")
    model.save(final_model_save_path)

    # 9. Shutdown Tora client
    tora.shutdown()
