from transformers import AutoImageProcessor, SiglipVisionModel
from PIL import Image
import torch
import numpy as np
import pydicom
import os
import pickle
import math
from collections import Counter
import time 

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on Device {}".format(device))

# Load model and processor
model = SiglipVisionModel.from_pretrained("google/medsiglip-448").to(device)
processor = AutoImageProcessor.from_pretrained("google/medsiglip-448", use_fast=False)

def return_model_embeddings_batch(images: list[Image.Image]):
    """
    Process a list of PIL images and return normalized embeddings as a NumPy array.
    """
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs["pooler_output"]
        embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
    return embeddings.cpu().numpy()

def filter_data(dcm_files, threshold=256):
    less_than_threshold = True if len(dcm_files) <= threshold else False
    return less_than_threshold

scan_folder = "tciaDownload"
output = []
lung_cancer_subtype = []

series_list = os.listdir(scan_folder)
class_a_samples = 100

for series in series_list:
    try:
        print(f"Starting series: {series}")
        series_path = os.path.join(scan_folder, series)

        if not os.path.isdir(series_path):
            continue

        dcm_files = sorted([f for f in os.listdir(series_path) if f.lower().endswith(".dcm")])
        if not dcm_files:
            continue

        less_than_threshold = filter_data(dcm_files)
        if not less_than_threshold:
            print("Series : {} greater than threshold 256".format(series))
            continue

        images = []
        last_ds = None
        array_build_start = time.perf_counter()
        for image_name in dcm_files:
            image_path = os.path.join(series_path, image_name)
            ds = pydicom.dcmread(image_path)
            last_ds = ds

            image = ds.pixel_array.astype(np.float32)

            slope = getattr(ds, "RescaleSlope", 1)
            intercept = getattr(ds, "RescaleIntercept", 0)
            image = image * slope + intercept

            image = Image.fromarray(image).convert("RGB")
            images.append(image)

        array_build_end = time.perf_counter()

        class_type = getattr(last_ds, "PatientID", None)
        if class_type is None:
            continue

        class_type = class_type.replace("Lung_Dx-", "")[0].lower()

        # --- enforce limit for class A BEFORE appending anything ---
        if class_type == "a":
            if class_a_samples <= 0:
                continue
            class_a_samples -= 1

        model_output_start = time.perf_counter()
        # compute embeddings only if we're keeping this sample
        series_embeddings = return_model_embeddings_batch(images)

        model_output_end = time.perf_counter()

        output.append(series_embeddings)
        lung_cancer_subtype.append(class_type)

        print(f"""
        Completed Series: {series}
        Class A sample Status: {class_a_samples}
        Classes: {Counter(lung_cancer_subtype)}
        Total Samples collected: {len(lung_cancer_subtype)}
        Total Embedding arrays: {len(output)}
        Time Taken to build the array for model : {array_build_end-array_build_start}
        Time Taken to process the series : {model_output_end-model_output_start}
        """)

    except Exception as e:
        continue

# Save results with pickle
with open("lung_cancer_subtype.pkl", "wb") as f:
    pickle.dump(lung_cancer_subtype, f)

with open("output_embeddings.pkl", "wb") as f:
    pickle.dump(output, f)

print("Pickle files saved successfully!")
