import os
import pickle
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


# Constants & Config
SAMPLES_FILE = "sft_data_for_medgemma.pkl"
MODEL_ID = "google/medgemma-4b-it"
OUTPUT_DIR = "medgemma-4b-it-sft-lora-lung-cancer-dt-mol"
NUM_TRAIN_EPOCHS = 2
LEARNING_RATE = 5e-4
BATCH_SIZE = 2
GRAD_ACCUMULATION = 1
EVAL_STEPS = 15


# Utility Functions
def load_samples(file_path: str) -> List[Dict]:
    """Load dataset samples from a pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def collate_fn(samples: List[Dict], processor: AutoProcessor) -> Dict:
    """
    Data collator for SFT training.
    Prepares prompts and images for the model and handles label masking.
    """
    prompts = []
    images_list = []

    for sample in samples:
        prompt = processor.apply_chat_template(
            sample["messages"],
            add_generation_prompt=False,
            tokenise=False
        ).strip()
        prompts.append(prompt)
        images_list.append([sample["image"]])

    batch = processor(text=prompts, images=images_list, return_tensors="pt", padding=True)

    labels = batch["input_ids"].clone()
    image_token_id = processor.tokenizer.convert_tokens_to_ids(
        processor.tokenizer.special_tokens_map["boi_token"]
    )
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100 
    batch["labels"] = labels
    return batch


def check_gpu_bf16_support():
    """Ensure the GPU supports bfloat16 precision."""
    if torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("GPU does not support bfloat16. Use an A100 or H100.")


# Main Script
def main():
    # Load dataset
    samples = load_samples(SAMPLES_FILE)
    train_samples, test_samples = train_test_split(samples, test_size=0.1, random_state=42)

    # Load processor
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "right"

    # Check GPU capabilities
    check_gpu_bf16_support()

    # Model kwargs for bfloat16 and 4-bit quantization
    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        ),
    )

    # Load model
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **model_kwargs)

    # LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=32,
        bias="none",
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj", "lm-head"],
        task_type="CAUSAL_LM",
    )

    # SFT training configuration
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=2,
        save_strategy="epoch",
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        learning_rate=LEARNING_RATE,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        push_to_hub=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        label_names=["labels"],
    )

    # Trainer initialization
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_samples,
        eval_dataset=test_samples,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=lambda batch: collate_fn(batch, processor),
    )

    # Start training
    trainer.train()
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
