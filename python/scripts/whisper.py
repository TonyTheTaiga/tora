import torch
import torchaudio
import aiohttp
from datasets import load_dataset
import evaluate
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForSeq2Seq,  # Correct for encoder-decoder models like Whisper
)
from tora import Tora
import numpy as np


# --- Configuration ---
MODEL_NAME = "openai/whisper-small"

# --- Option 2: Common Voice 11 (Uncomment and adjust for Common Voice) ---
DATASET_NAME = "mozilla-foundation/common_voice_11_0"
DATASET_CONFIG = "en"  # Choose your desired language, e.g., "en" for English
DATASET_SPLIT_TRAIN = "train[:5000]"  # Common Voice can be larger, adjust as needed
DATASET_SPLIT_EVAL = "validation[:500]"


OUTPUT_DIR = (
    "./whisper-small-fine-tuned-common-voice"  # Updated output directory for clarity
)
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-4
NUM_TRAIN_EPOCHS = 3
SAVE_STEPS = 500
EVAL_STEPS = 500
LOGGING_STEPS = 50
TORA_EXPERIMENT_NAME = (
    "Whisper-Small Fine-tuning (Common Voice)"  # Updated experiment name
)
TORA_DESCRIPTION = "Fine-tuning Whisper-Small on Common Voice 11.0 English dataset."
TORA_TAGS = ["whisper", "common-voice", "ASR", "fine-tuning"]


# --- Data Preprocessing ---
# Modify prepare_dataset to accept processor as an argument
def prepare_dataset(batch, processor):
    audio = batch["audio"]
    # Resample to 16kHz if necessary
    if audio["sampling_rate"] != 16000:
        resampler = torchaudio.transforms.Resample(audio["sampling_rate"], 16000)
        audio_array = resampler(
            torch.tensor(audio["array"], dtype=torch.float32)
        ).numpy()
    else:
        audio_array = audio["array"]

    # Process audio and text using WhisperProcessor
    # The processor returns 'input_features' (log-Mel spectrograms) and 'labels' (tokenized text).
    # It also handles padding/clipping to 30 seconds internally for Whisper.
    model_inputs = processor(
        audio=audio_array,
        sampling_rate=16000,  # Always pass 16kHz to the processor after resampling
        text=batch["sentence"],  # Use 'sentence' for Common Voice, not 'text'
    )

    # The Trainer expects 'input_features' and 'labels'
    batch["input_features"] = model_inputs.input_features[
        0
    ]  # Access the tensor from the list
    batch["labels"] = model_inputs.labels[0]  # Access the tensor from the list

    return batch


# --- Metric Calculation ---
wer_metric = evaluate.load("wer")


def compute_metrics(pred):
    # pred.predictions are the generated token IDs when predict_with_generate=True
    # pred.label_ids are the true token IDs

    # Decode predictions and labels, skipping special tokens for WER calculation
    pred_str = processor.batch_decode(pred.predictions, skip_special_tokens=True)
    # Replace -100 in label_ids with pad_token_id before decoding
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(pred.label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# --- Main Training Script ---
if __name__ == "__main__":
    # 1. Initialize Tora Experiment
    tora = Tora.create_experiment(
        name=TORA_EXPERIMENT_NAME,
        description=TORA_DESCRIPTION,
        hyperparams={
            "model_name": MODEL_NAME,
            "dataset_name": DATASET_NAME,
            "language": DATASET_CONFIG,  # Log the language for Common Voice
            "learning_rate": LEARNING_RATE,
            "num_train_epochs": NUM_TRAIN_EPOCHS,
            "per_device_train_batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "train_split": DATASET_SPLIT_TRAIN,
            "eval_split": DATASET_SPLIT_EVAL,
        },
        tags=TORA_TAGS,
    )
    print(f"Tora Experiment created with ID: {tora._experiment_id}")

    # 2. Load Dataset
    print(f"Loading dataset: {DATASET_NAME} - {DATASET_CONFIG}")
    common_voice_train = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=DATASET_SPLIT_TRAIN,
        token=True,  # Add token=True if you encounter issues with Common Voice dataset access
        storage_options={
            "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}
        },
    )
    common_voice_test = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=DATASET_SPLIT_EVAL,
        token=True,  # Add token=True
        storage_options={
            "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}
        },
    )

    # Common Voice datasets usually have a "sentence" column for text, not "text"
    # Make sure 'sentence' is the correct column name for the text.

    # Filter out samples with too long audio to avoid memory issues and speed up
    # Whisper models are designed to process audio up to 30 seconds.
    MAX_AUDIO_LENGTH_SAMPLES = 16_000 * 30  # Max 30 seconds for Whisper

    # Removed the filter based on x["audio"]["sampling_rate"] == 16000 here.
    # The prepare_dataset function handles resampling for all audio.
    common_voice_train = common_voice_train.filter(
        lambda x: len(x["audio"]["array"]) < MAX_AUDIO_LENGTH_SAMPLES
    )
    common_voice_test = common_voice_test.filter(
        lambda x: len(x["audio"]["array"]) < MAX_AUDIO_LENGTH_SAMPLES
    )

    print(
        f"Train dataset size after initial length filtering: {len(common_voice_train)}"
    )
    print(f"Eval dataset size after initial length filtering: {len(common_voice_test)}")

    # 3. Load Pre-trained Model and Processor
    print(f"Loading processor and model: {MODEL_NAME}")
    # Initialize processor here, it will be passed to prepare_dataset
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    # Move model to GPU if available
    if torch.cuda.is_available():
        model.to("cuda")

    # For Whisper, the decoder_start_token_id is important for generation
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.forced_decoder_ids = (
        None  # Set to None for general ASR, or specify for multilingual tasks
    )
    model.generation_config.lang_as_token = (
        True  # Enable language token if fine-tuning multilingual Whisper
    )
    model.generation_config.task = "transcribe"  # Task token for transcription

    # Apply preprocessing to dataset
    print("Preprocessing datasets...")
    # Pass processor to prepare_dataset using fn_kwargs
    common_voice_train = common_voice_train.map(
        prepare_dataset,
        fn_kwargs={"processor": processor},  # Pass processor here
        remove_columns=common_voice_train.column_names,
        num_proc=4,  # Adjust based on your CPU cores
        batched=False,  # Crucial for processing single examples
    ).filter(
        # Filter out any empty audio after processing.
        # Check if 'input_features' has a '.size' attribute (for numpy arrays/tensors)
        # and if its total number of elements is greater than 0.
        lambda x: hasattr(x["input_features"], "size") and x["input_features"].size > 0
    )

    common_voice_test = common_voice_test.map(
        prepare_dataset,
        fn_kwargs={"processor": processor},  # Pass processor here
        remove_columns=common_voice_test.column_names,
        num_proc=4,  # Adjust based on your CPU cores
        batched=False,  # Crucial for processing single examples
    ).filter(
        lambda x: hasattr(x["input_features"], "size") and x["input_features"].size > 0
    )

    print(
        f"Train dataset size after preprocessing & filtering: {len(common_voice_train)}"
    )
    print(
        f"Eval dataset size after preprocessing & filtering: {len(common_voice_test)}"
    )

    # 4. Define Data Collator
    # DataCollatorForSeq2Seq is suitable for encoder-decoder models like Whisper
    # It requires the tokenizer and model's pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        processor.tokenizer,
        model=model,
        padding=True,
        return_tensors="pt",
    )

    # 5. Configure Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        group_by_length=False,  # Set to False for DataCollatorForSeq2Seq with Whisper
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        eval_strategy="steps",  # Changed from eval_strategy to evaluation_strategy
        num_train_epochs=NUM_TRAIN_EPOCHS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        logging_steps=LOGGING_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,  # Keep all columns for the data collator to work
    )

    # 6. Create Trainer
    print("Initializing Trainer...")

    class ToraCallback(TrainerCallback):
        def __init__(self, tora_client: Tora):
            self.tora_client = tora_client

        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_world_process_zero:
                if logs is not None:
                    for key, value in logs.items():
                        if isinstance(value, (int, float)) and not key.endswith(
                            "_total_loss"
                        ):
                            self.tora_client.log(
                                name=key, value=value, step=state.global_step
                            )

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if state.is_world_process_zero:
                if metrics is not None:
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            self.tora_client.log(
                                name=key, value=value, step=state.global_step
                            )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=common_voice_train,
        eval_dataset=common_voice_test,
        processing_class=processor.tokenizer,  # Pass the tokenizer for decoding during evaluation
    )

    trainer.add_callback(ToraCallback(tora))

    # 7. Start Training
    print("Starting training...")
    trainer.train()

    # 8. Save the fine-tuned model
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)  # Save the processor as well

    # 9. Shutdown Tora client to ensure all logs are written
    tora.shutdown()
    print("Training complete and Tora client shut down.")
