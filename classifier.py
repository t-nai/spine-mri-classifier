import os
import csv
import numpy as np
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm
import google.generativeai as genai
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import pandas as pd

genai.configure(api_key = "your api key here") # user must modify this line

model = genai.GenerativeModel("gemini-2.5-pro-preview-06-05")

image_root_dir = "your images folder file path here" # user must modify this line

ground_truth_csv = "your file path to groundtruth.csv here" # user must modify this line

gt_df = pd.read_csv(ground_truth_csv)
gt_dict = dict(zip(gt_df['image_file'], gt_df['Herniation_Label']))

def classify_with_gemini(image_path):
    try:
        sitk_image = sitk.ReadImage(image_path)
        array = sitk.GetArrayFromImage(sitk_image)

        if array.ndim == 3:
            slice_2d = array[array.shape[0] // 2]
        else:
            slice_2d = array

        slice_2d = (255 * (slice_2d - np.min(slice_2d)) / (np.ptp(slice_2d) + 1e-5)).astype(np.uint8)
        pil_image = Image.fromarray(slice_2d)

        prompt = (
            "This is a sagittal view of a lumbar spine MRI from a patient.\n"
            "Based on the image, is there evidence of a disc herniation?\n\n"
            "A) One or more disc herniation(s) are present\n"
            "B) No disc herniation is present.\n\n"
            "Answer this question with a single letter ONLY."
        )

        response = model.generate_content([prompt, pil_image])
        raw_output = response.text.strip()
        print(f"Raw output for {os.path.basename(image_path)}: {raw_output}")

        if raw_output == "A":
            return "A"
        elif raw_output == "B":
            return "B"
        elif "A" in raw_output:
            return "A"
        elif "B" in raw_output:
            return "B"
        else:
            return "None"
    except Exception as e:
        print(f"Error classifying {image_path}: {e}")
        return "None"

data = []

for fname in os.listdir(image_root_dir):
    fname_lower = fname.lower()
    if fname_lower.endswith(".mha") and "t1" in fname_lower and not ("t2" in fname_lower or "t2_space" in fname_lower):
        base_name = os.path.splitext(fname)[0]
        gt_label = gt_dict.get(base_name, "?")
        img_path = os.path.join(image_root_dir, fname)
        data.append((img_path, gt_label))

results = []

print(f"Classifying {len(data)} MRI images with Gemini...")

for img_path, ground_truth in tqdm(data):
    prediction = classify_with_gemini(img_path)
    if prediction is None:
        prediction = "None"
    results.append((img_path, ground_truth, prediction))

valid_labels = {"A", "B"}
y_true_filtered = [gt for _, gt, pred in results if pred in valid_labels and gt in valid_labels]
y_pred_filtered = [pred for _, gt, pred in results if pred in valid_labels and gt in valid_labels]

if len(y_true_filtered) > 0:
    tn, fp, fn, tp = confusion_matrix(y_true_filtered, y_pred_filtered, labels=["A", "B"]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
    f1 = f1_score(y_true_filtered, y_pred_filtered, pos_label="A")

    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
else:
    print("Ground truth labels not available — skipping metrics.")

with open("gemini_spine_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image", "Ground Truth", "Prediction"])
    writer.writerows(results)