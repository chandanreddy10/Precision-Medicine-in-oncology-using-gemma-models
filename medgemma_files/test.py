import torch
from PIL import Image
import requests
from transformers import AutoModelForImageTextToText, AutoProcessor
import os
import pickle

torch._dynamo.config.disable = True

MODEL_PATH = "medgemma-4b-it-sft-lora-lung-cancer/checkpoint-129"  

# Automatically set device and data type
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Use bfloat16 if supported (on Ampere GPUs like A100), otherwise float16
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

print(f"Using device: {DEVICE}")
print(f"Using dtype: {DTYPE}")

# --- Load Model & Processor ---
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    device_map="auto",  # Automatically handle model placement on devices
    trust_remote_code=True  # Add this if needed for custom model code
)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = processor.tokenizer

with open("sft_data_for_medgemma.pkl","rb") as file:
    formatted_data = pickle.load(file)

image =formatted_data[0]['image']

# The prompt for the model
user_prompt = "Analyse this CT Scan and provide step-by-step findings. Identify the cancer subtype and other information from the CT Scan. Give me a detailed explanation."

chat = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": user_prompt}
        ],
    }
]
formatted_prompt = processor.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

# --- Run Inference ---
# Process the text and image together
inputs = processor(text=formatted_prompt, images=image, return_tensors="pt").to(DEVICE)

# Move inputs to correct dtype if needed
if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
    inputs.pixel_values = inputs.pixel_values.to(dtype=DTYPE)

input_ids_len = inputs["input_ids"].shape[-1]

# Generate a response from the model with additional safeguards
with torch.inference_mode():
    try:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=500,
            use_cache=True,
            do_sample=False,  # Use greedy decoding for more stable results
            pad_token_id=tokenizer.eos_token_id,  # Explicitly set pad token
            temperature=0.7,  # Add temperature control
            top_p=0.9,  # Add nucleus sampling
        )
    except Exception as e:
        print(f"Error during generation: {e}")
        print("Trying with simplified generation parameters...")
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs.get("pixel_values"),
            max_new_tokens=200,
            pad_token_id=tokenizer.eos_token_id,
        )

# Decode the generated tokens to text, skipping the prompt
response = processor.decode(output_ids[0, input_ids_len:], skip_special_tokens=True)

# --- Output ---
print("\n📌 Model Prediction:")
print(response)
