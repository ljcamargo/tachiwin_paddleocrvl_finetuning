# Tachiwin-OCR: Fine-tuning PaddleOCR-VL for Indigenous Languages

Tachiwin-OCR is a specialized project dedicated to improving Optical Character Recognition (OCR) for the Indigenous languages of Mexico. This repository provides a complete pipeline for generating synthetic datasets, fine-tuning, and evaluating high-accuracy OCR models for under-represented languages.

**Note:** This project focuses on the training and evaluation workflow. It is a fine-tune of the [PaddleOCR-VL](https://huggingface.co/unsloth/PaddleOCR-VL) architecture. This repository does not provide a direct deployment or production-serving method.

---

## 🚀 Key Resources

- **Model Weights:** [tachiwin/PaddleOCR-VL-Tachiwin](https://huggingface.co/tachiwin/PaddleOCR-VL-Tachiwin)
- **Dataset:** [tachiwin/multilingual_ocr_llm_2](https://huggingface.co/datasets/tachiwin/multilingual_ocr_llm_2)

---

## 📂 Project Structure

### 1. Dataset Generation & Uploading
- **`generator_llm.py`**: A robust synthetic data generator that converts text corpora into high-quality training images. Features include:
    - Support for multiple fonts (Andika, DejaVu, MPLUS, etc.) that handle special glyphs for Indigenous languages.
    - Advanced distortions: Skew, perspective, noise, blur, banding, and morphology effects.
    - Deterministic parameter selection for reproducibility across runs.
- **Uploader Scripts**: 
    - `dataset_uploader_llm.py`: Main script for uploading image-text pairs to Hugging Face.
    - `dataset_uploader_llm_preprocessed.py`: Optimized for pre-processed datasets.
    - `dataset_uploader_erniesdk.py`: Integration with ErnieSDK for specific data flows.
    - Includes sharding, resume capabilities, and retry logic to handle large-scale uploads.

### 2. Model Fine-tuning
- **`Tachiwin_OCR_PaddleOCR_VL_Finetuning.ipynb`**: The primary notebook for model training. It uses **Unsloth** to achieve significantly faster training and lower memory usage. It is specifically optimized for the PaddleOCR-VL architecture.

### 3. Inference
You can perform inference using the `PaddleOCR` pipeline or the `transformers` library.

#### Option A: Using PaddleOCR (Easy Pipeline)
```python
from paddleocr import PaddleOCRVL

# Load the fine-tuned model
pipeline = PaddleOCRVL(
    vl_rec_model_name="PaddleOCR-VL-0.9B",
    vl_rec_model_dir=model_dir,
)

# Predict on an image
output = pipeline.predict("test.png")

for res in output:
    res.print()
    res.save_to_json(save_path="output")
    res.save_to_markdown(save_path="output")
```

#### Option B: Using Transformers (Advanced Control)
```python
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

# ---- Settings ----
model_path = "tachiwin/PaddleOCR-VL-Tachiwin"
image_path = "test.png"
task = "ocr" # Options: 'ocr' | 'table' | 'chart' | 'formula'
# ------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}

image = Image.open(image_path).convert("RGB")

model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

messages = [
    {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": PROMPTS[task]},
    ]}
]

inputs = processor.apply_chat_template(
    messages, 
    tokenize=True, 
    add_generation_prompt=True, 	
    return_dict=True,
    return_tensors="pt"
).to(DEVICE)

outputs = model.generate(**inputs, max_new_tokens=1024)
generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print(generated_text)
```

### 4. Evaluation
- **`ocr_evaluator.py`**: A lightweight tool to measure model performance.
    - **Metrics:** Computes Character Error Rate (CER) and Word Error Rate (WER).
    - **Comparison:** Directly compares "Raw" model results against "Fine-tuned" results using a ground truth JSON.
    - **Usage:** 
      ```bash
      python ocr_evaluator.py samples.json
      ```

---

## 🛠️ Installation & Setup

### Requirements
- Python 3.10+
- PyTorch (for inference/finetuning)
- `pip install unsloth` (for the finetuning notebook)
- `pip install datasets huggingface_hub PIL transformers`

### Usage Flow
1. **Generate Data:** Use `generator_llm.py` to create a synthetic dataset from your text files.
2. **Upload:** Use the uploader scripts to push your data to Hugging Face.
3. **Fine-tune:** Run the `Tachiwin_OCR_PaddleOCR_VL_Finetuning.ipynb` on a GPU (Colab/RunPod recommended).
4. **Evaluate:** Use `ocr_evaluator.py` to compare performance gains.

---

## 📈 Goals
Traditional OCR models often fail on Indigenous languages due to unique character sets and lack of training data. Tachiwin-OCR bridges this gap by creating high-quality synthetic data that mimics real-world document challenges (poor scans, diverse fonts, complex layouts) and fine-tuning state-of-the-art vision models to understand them.

---

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Developed with ❤️ by the Tachiwin project.
