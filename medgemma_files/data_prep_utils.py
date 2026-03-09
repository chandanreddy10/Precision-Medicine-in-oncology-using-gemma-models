import os
import json
import logging

import pandas as pd
import pydicom
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup

from ollama import chat

#Setting up log config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    filename="data_prep.log"
)

# Filepaths and constants
XML_FILES_DIR = "AIM_files_updated-11-10-2020"
OUTPUT_JSON_FILE = "data_from_xml.json"
CSV_FILE = "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"
DATA_GEN_PROMPT_FILE = "prompt_for_data_generation.txt"
MEDGEMMA_USER_PROMPT_FILE = "medgemma_user_prompt.txt"
CT_SCAN_ROOT = "tciaDownload"

# Load prompts
with open(DATA_GEN_PROMPT_FILE, "r") as f:
    DATA_GEN_PROMPT = f.read()

with open(MEDGEMMA_USER_PROMPT_FILE, "r") as f:
    MEDGEMMA_USER_PROMPT = f.read()


# Helper Functions
def build_sample(image: Image.Image, user_prompt: str, assistant_response: str) -> dict:
    """Constructs SFT data sample with image + conversation messages."""
    return {
        "image": image,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_response}
                ]
            }
        ]
    }


def gemma_call(data: dict, prompt: str = DATA_GEN_PROMPT) -> str:
    """Call Gemma3 model to generate a response for SFT."""
    message = f"{prompt} Data: {data}"
    response = chat(
        model='gemma3:12b',
        messages=[{'role': 'user', 'content': message}],
    )
    return response.message.content


def crack_format(series_path: str, frame_number: str) -> str | None:
    """
    Attempts multiple filename zero-padded patterns to locate DICOM file.
    Returns the first valid path found or None.
    """
    patterns = [frame_number, frame_number.zfill(2), frame_number.zfill(3),
                frame_number.zfill(4), frame_number.zfill(5)]
    for pattern in patterns:
        file_path = os.path.join(series_path, f"1-{pattern}.dcm")
        if os.path.exists(file_path):
            return file_path
    return None


def return_image_object(dcm_file_path: str, size=(224, 224)) -> Image.Image | None:
    """
    Reads a DICOM file, converts to a 224x224 RGB PIL Image.
    Returns None if the image cannot be handled.
    """
    try:
        ds = pydicom.dcmread(dcm_file_path)
        image = ds.pixel_array.astype(np.float32)

        # Apply slope/intercept
        slope = getattr(ds, "RescaleSlope", 1)
        intercept = getattr(ds, "RescaleIntercept", 0)
        image = image * slope + intercept

        # Remove singleton dimensions
        if image.ndim > 2:
            image = np.squeeze(image)

        # Normalize to 0-255
        min_val, max_val = np.min(image), np.max(image)
        if max_val - min_val == 0:
            image = np.zeros_like(image, dtype=np.uint8)
        else:
            image = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        # Convert grayscale to RGB
        if image.ndim == 2:
            image = np.stack([image]*3, axis=-1)

        return Image.fromarray(image).resize(size)

    except Exception as e:
        logging.warning(f"Skipping DICOM {dcm_file_path} due to error: {e}")
        return None

def locate_the_series_for_key(patient_key: str, ct_scan_root: str) -> tuple[str | None, str | None]:
    """
    Locate the CT series directory and confirm patient ID matches.
    Returns (series_path, patient_id) or (None, None) if not found.
    """
    for series in os.listdir(ct_scan_root):
        series_path = os.path.join(ct_scan_root, series)
        dcm_files = sorted([f for f in os.listdir(series_path) if f.lower().endswith(".dcm")])
        if not dcm_files:
            continue

        try:
            ds = pydicom.dcmread(os.path.join(series_path, dcm_files[0]))
            patient_id = getattr(ds, "PatientID", None)
            if patient_id == patient_key:
                return series_path, patient_id
        except Exception as e:
            logging.warning(f"Failed to read DICOM {dcm_files[0]}: {e}")
    return None, None


# XML Parsing & JSON Generation
def get_data_from_xml_files(xml_dir=XML_FILES_DIR, output_file: str = OUTPUT_JSON_FILE) -> None:
    """Parse all XML files and save structured data to JSON."""
    data_from_xml = {}

    for idx, xml_file_name in enumerate(os.listdir(xml_dir), start=1):
        xml_file_path = os.path.join(xml_dir, xml_file_name)
        logging.info(f"Processing XML file {idx}/{len(os.listdir(xml_dir))}: {xml_file_name}")

        characteristic_key = "iso:displayName" if xml_file_name[0].upper() == "R" else "typeCode"
        characteristic_value = "value" if xml_file_name[0].upper() == "R" else "codeSystem"

        with open(xml_file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), features="lxml-xml")

        person = soup.find("person").find("id").get("value")
        sample = {}

        # Imaging physical entity
        physical_entity = soup.find("imagingPhysicalEntityCollection")
        loc_value = physical_entity.find(characteristic_key).get(characteristic_value)
        loc_key = physical_entity.find("label").get("value")
        sample[loc_key] = loc_value

        # Imaging observations
        observation_entities = soup.find("imagingObservationEntityCollection").find_all("ImagingObservationEntity")
        characteristics = []
        for entity in observation_entities:
            characteristics = entity.find_all("ImagingObservationCharacteristic")
            if characteristics:
                break

        for char in characteristics:
            key = char.find("label").get("value")
            value = char.find(characteristic_key).get(characteristic_value)
            sample[key] = value

        # Markup entity for frame number
        markup_entity = soup.find("markupEntityCollection")
        frame_number = markup_entity.find("referencedFrameNumber").get("value")
        sample["image_frame_number"] = frame_number

        data_from_xml[person] = sample
        logging.info(f"Completed Person {person} | Samples so far: {idx}")

    with open(output_file, "w") as f:
        json.dump(data_from_xml, f, indent=2)
    logging.info(f"All XML files processed. Output saved to {output_file}")


def add_histology_info(csv_file: str, output_file: str = OUTPUT_JSON_FILE) -> None:
    """Add Histology info from CSV to JSON data."""
    with open(output_file, "r") as f:
        contents = json.load(f)

    df = pd.read_csv(csv_file)
    group_patients = df.groupby("Case ID")
  
    columns = [
        "Weight (lbs)", "Gender","Smoking status", "Pack Years",
        "%GG", "Tumor Location (choice=RUL)",
        "Tumor Location (choice=RML)", "Tumor Location (choice=RLL)",
        "Tumor Location (choice=LUL)", "Tumor Location (choice=LLL)",
        "Tumor Location (choice=L Lingula)", "Tumor Location (choice=Unknown","Histology ", "Pathological T stage", "Pathological N stage",
        "Pathological M stage", "Histopathological Grade",
        "Lymphovascular invasion", "Pleural invasion (elastic, visceral, or parietal)","EGFR mutation status", "KRAS mutation status", "ALK translocation status"
    ]

    for patient_id, description in contents.items():
        if patient_id not in group_patients.groups:
            logging.warning(f"No patient group found for patient {patient_id}")
            continue

        patient_group = group_patients.get_group(patient_id)
        for col in columns:
            clean_col = col.strip()
            value = patient_group[col].values[0] if col in patient_group.columns else None
            description[clean_col] = value

        logging.info(f"Added clinical info for patient {patient_id}")

    with open(output_file, "w") as f:
        json.dump(contents, f, indent=2)
    logging.info("JSON file updated with histology info.")


# Example Usage: Generate SFT Samples
if __name__ == "__main__":
    # Load JSON with annotations
    with open(OUTPUT_JSON_FILE, "r") as f:
        contents = json.load(f)

    # List all series directories in CT scan root
    ct_scan_series = os.listdir(CT_SCAN_ROOT)

    for key, description in contents.items():
        series_path, patient_id = locate_the_series_for_key(key, CT_SCAN_ROOT)
        if not series_path:
            logging.warning(f"Series not found for patient {key}")
            continue

        frame_number = description.get("image_frame_number")
        if not frame_number:
            logging.warning(f"No frame number for patient {key}")
            continue

        file_path = crack_format(series_path, frame_number)
        if not file_path:
            logging.warning(f"DICOM file not found for patient {key}, frame {frame_number}")
            continue

        # Process image
        image = return_image_object(file_path)

        # Generate assistant response via Gemma3
        assistant_response = gemma_call(description, DATA_GEN_PROMPT)

        # Build SFT sample
        sample = build_sample(image, MEDGEMMA_USER_PROMPT, assistant_response)

        logging.info(f"SFT sample created for patient {key}")

        # For demonstration, only process the first patient
        break
