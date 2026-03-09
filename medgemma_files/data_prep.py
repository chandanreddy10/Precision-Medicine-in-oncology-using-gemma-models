# from bs4 import BeautifulSoup
# import json 
# import os 
# import pandas as pd
# from ollama import chat
# import pydicom
# from PIL import Image
# import numpy as np
# import logging 

# xml_files_dir = "AIM_files_updated-11-10-2020"
# output_file = "data_from_xml.json"
# csv_file = "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"
# data_gen_text_file = "prompt_for_data_generation.txt"
# medgemma_user_prompt_file = "medgemma_user_prompt.txt"
# ct_scan_rf = "tciaDownload"
# ct_scan_series = os.listdir(ct_scan_rf)

# with open(data_gen_text_file, "r") as file:
#     data_gen_prompt = file.read()

# with open(medgemma_user_prompt_file, "r") as file:
#     medgemma_user_prompt = file.read()

# def build_sample(image, user_prompt, assistant_response):
#     return {
#     "image":image,
#     "messages": [
#         {
#         "role": "user",
#         "content": [
#             {"type": "image"},
#             {"type": "text", "text": user_prompt}
#         ]
#         },
#         {
#         "role": "assistant",
#         "content": [
#             {"type": "text", "text": assistant_response}
#         ]
#         }
#     ]
# }

# def gemma_call(data, prompt=data_gen_prompt):
#     message = f"{prompt} Data : {data}"
#     response = chat(
#         model='gemma3:12b',
#         messages=[{'role': 'user', 'content': message}],
#     )
#     return response.message.content

# def crack_format(series_path, frame_number):
#     patterns = [frame_number, frame_number.zfill(2),frame_number.zfill(3), frame_number.zfill(4), frame_number.zfill(5)]
#     file_path_result = None
#     for pattern in patterns:
#         file_path = os.path.join(series_path, f"1-{pattern}.dcm")
#         if os.path.exists(file_path):
#             file_path_result = file_path
#             break
#     return file_path_result

# def get_data_from_xml_files_dir(xml_files_dir, output_file=output_file):

#     data_from_xml = {}

#     for indx, xml_file_name in enumerate(os.listdir(xml_files_dir)):
#         characteristic_key ="iso:displayName"  if xml_file_name[0].upper() == "R" else "typeCode"
#         characteristic_value = "value" if xml_file_name[0].upper() == "R" else "codeSystem"
#         print(characteristic_key, characteristic_value)
#         print("Starting XML file : {}".format(xml_file_name))
#         xml_file = os.path.join(xml_files_dir, xml_file_name)
#         with open(xml_file, "r", encoding="utf-8") as f:
#             xml_content = f.read()

#         soup = BeautifulSoup(xml_content, features="lxml-xml")
#         person = soup.find("person").find("id").get("value")

#         sample = {}
#         physical_entity = soup.find("imagingPhysicalEntityCollection")

#         loc_value = physical_entity.find(characteristic_key).get(characteristic_value)
#         loc_key = physical_entity.find("label").get("value")

#         observation_entity = soup.find("imagingObservationEntityCollection").find("ImagingObservationEntity")

#         characteristics = observation_entity.find_all("ImagingObservationCharacteristic")
#         if not characteristics:
#             observation_entity = soup.find("imagingObservationEntityCollection").find_all("ImagingObservationEntity")
#             for entity in observation_entity:
#                 characteristics = entity.find_all("ImagingObservationCharacteristic")
#                 if characteristics :
#                     break
#         sample.update({loc_key:loc_value})

#         for characteristic in characteristics:
#             value =characteristic.find(characteristic_key).get(characteristic_value)
#             key =characteristic.find("label").get("value")
#             sample.update({key:value})

#         markup_entity = soup.find("markupEntityCollection")
#         image_frame_number = markup_entity.find("referencedFrameNumber").get("value")
#         sample.update({"image_frame_number":image_frame_number})

#         data_from_xml.update({person:sample})
#         print("Completed Person {} | Data Sample Count : {}".format(person, indx+1))

#     with open(output_file, "w") as file:
#         json.dump(data_from_xml, file, indent=2)
#     print("Dumped Results to Json.")

# def add_histology_info(csv_file, output_file):
#     with open(output_file, "r") as file:
#         contents = json.load(file)
#     df = pd.read_csv(csv_file)
#     group_patients = df.groupby("Case ID")
#     for key, description in contents.items():
#         print("Patient ID {}".format(key))
#         description["Histology"] = group_patients.get_group(key)["Histology "].values[0]
#         print("Patient ID {} Done.".format(key))
#     with open(output_file, "w") as file:
#         json.dump(contents,file, indent=2)
#     print("Updated the JSON file with cancer subtype")

# def locate_the_series_for_key(key, ct_scan_rf, ct_scan_series):
#     series_path, patient_id = None, None
#     for series in ct_scan_series:
#         series_path = os.path.join(ct_scan_rf, series)
#         dcm_files = sorted([f for f in os.listdir(series_path) if f.lower().endswith(".dcm")])

#         file_path = os.path.join(series_path, dcm_files[0])

#         file_data = pydicom.dcmread(file_path)
#         patient_id = getattr(file_data, "PatientID", None)
#         if patient_id and patient_id == key:
#             break
#     return series_path, patient_id

# def return_image_object(dcm_file_path):
#     ds = pydicom.dcmread(dcm_file_path)
#     image = ds.pixel_array.astype(np.float32)

#     slope = getattr(ds, "RescaleSlope", 1)
#     intercept = getattr(ds, "RescaleIntercept", 0)
#     image = image * slope + intercept

#     image_min = np.min(image)
#     image_max = np.max(image)
#     image = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)

#     # Convert to RGB
#     image = Image.fromarray(image).convert("RGB")

#     # Resize to 224x224
#     image_resized = image.resize((224, 224))

#     return image_resized
# ## Extract the CT Scan Annotations from XML file.
# # get_data_from_xml_files_dir(xml_files_dir, output_file)

# ## Add the Cancer subtype information to the JSON file.
# # add_histology_info(csv_file, output_file)

# ## Use gemma3 to generate SFT data using CT scan Annotations and CT Scan Images.
# with open(output_file, "r") as file:
#     contents = json.load(file)

# for key, description in contents.items():
#     ct_scan_path, patient_id = locate_the_series_for_key(key, ct_scan_rf, ct_scan_series)
#     if ct_scan_path:
#         print(ct_scan_path, key, patient_id)
#         frame_number = description["image_frame_number"]
#         file_path = crack_format(ct_scan_path, frame_number)
#         if file_path:
#             image = return_image_object(file_path)
#             print(image)
#             assistant_response = gemma_call(description, data_gen_prompt)
#             sample = build_sample(image, medgemma_user_prompt, assistant_response)
#             print(sample)
#         else:
#             pass
#     break


import pickle
import logging
import json
import os

from data_prep_utils import OUTPUT_JSON_FILE, CT_SCAN_ROOT, DATA_GEN_PROMPT, MEDGEMMA_USER_PROMPT
from data_prep_utils import locate_the_series_for_key, crack_format, return_image_object, gemma_call, build_sample, add_histology_info, get_data_from_xml_files

OUTPUT_SAMPLE_FILE = "sft_data_for_medgemma.pkl"
CSV_FILE = "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"

get_data_from_xml_files()
add_histology_info(CSV_FILE, OUTPUT_JSON_FILE)
with open(OUTPUT_JSON_FILE, "r") as f:
    contents = json.load(f)

# List all series directories in CT scan root
ct_scan_series = os.listdir(CT_SCAN_ROOT)

samples = []
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
        logging.warning(f"DICOM file not found for patient {key}, frame {frame_number}, Series : {series_path}")
        continue

    # Process image
    image = return_image_object(file_path)
    if image is not None:
        # Generate assistant response via Gemma3
        assistant_response = gemma_call(description, DATA_GEN_PROMPT)

        # Build SFT sample
        sample = build_sample(image, MEDGEMMA_USER_PROMPT, assistant_response)
        logging.info(f"""
                    Patient Id : {key}
                    Series : {series_path}
                    Frame Number : {frame_number}
                    File Path : {file_path}
                    Gemma3 Response :{assistant_response}""")
        logging.info(f"SFT sample created for patient {key}")

        # For demonstration, only process the first patient
        samples.append(sample)
        logging.info(f"Sample Size : {len(samples)} | Total Data Points : {len(contents)}")

logging.info(f"Writing the Samples to {OUTPUT_SAMPLE_FILE}")
with open(OUTPUT_SAMPLE_FILE, "wb") as file:
    pickle.dump(samples, file)
logging.info(f"Dumped the Samples to {OUTPUT_SAMPLE_FILE}")