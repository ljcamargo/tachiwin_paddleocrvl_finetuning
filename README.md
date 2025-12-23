# Tachiwin-OCR: A Fine-tune of PaddleOCR-VL for Indigenous Languages

Tachiwin-OCR is a fine-tune of the [PaddleOCR-VL](https://huggingface.co/unsloth/PaddleOCR-VL) vision-language model, specifically developed to solve OCR challenges for the Indigenous languages of Mexico. This repository contains the code used to generate the specialized datasets and successfully fine-tune the model, serving as the source for reproducing the [Tachiwin-OCR weights](https://huggingface.co/tachiwin/PaddleOCR-VL-Tachiwin).

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
- **`Tachiwin_OCR_PaddleOCR_VL_Finetuning.ipynb`**: The implementation of the fine-tuning process. It leverages **Unsloth** for efficient training and is the code used to create Tachiwin-OCR by fine-tuning PaddleOCR-VL on the custom synthetic datasets.

### 3. Inference
You can perform inference using the `PaddleOCR` pipeline or the `transformers` library.

#### Option A: Using PaddleOCR (Easy Pipeline)
```python
from paddleocr import PaddleOCRVL

# Load the fine-tuned model
pipeline = PaddleOCRVL(
    vl_rec_model_name="PaddleOCR-VL-0.9B",
    vl_rec_model_dir=path_to_tachiwin_downloaded_model,
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
model_path = "tachiwin/PaddleOCR-VL-Tachiwin-BF16"
image_path = "test.png"
# ------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

image = Image.open(image_path).convert("RGB")

model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

messages = [
    {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "OCR:"},
    ]}
]

inputs = processor.apply_chat_template(
    messages, 
    tokenize=True, 
    add_generation_prompt=True, 	
    return_dict=True,
    return_tensors="pt"
).to(DEVICE)

outputs = model.generate(**inputs, max_new_tokens=1024, min_new_tokens=1)
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

## 📊 Benchmark Results

Tachiwin-OCR was evaluated against the base PaddleOCR-VL model using a diverse subset of Indigenous language samples. The fine-tuning results demonstrate significant improvements in both character and word recognition accuracy.

### Summary Metrics

| Metric | Base Model (Raw) | Tachiwin-OCR (Fine-tuned) | Improvement |
| :--- | :---: | :---: | :---: |
| **Character Error Rate (CER)** | 7.59% | 6.80% | **10.4% (Relative Reduction)** |
| **Word Error Rate (WER)** | 25.17% | 17.36% | **+7.81% (Absolute)** |
| **OCR Accuracy (1 - CER)** | 92.41% | 93.20% | **+0.79% (Absolute)** |

### Detailed Comparison (Sample)

A subset of the evaluation results across different languages, where tonal languages are the most improved by this fine-tuning:

| Language | Raw CER | FT CER | Raw WER | FT WER | Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `stp` (Tepehuán) | 10.95% | 0.00% | 43.55% | 0.00% | +10.95% |
| `maz` (Central Mazahua) | 3.29% | 0.41% | 9.09% | 0.00% | +2.88% |
| `chj` (Ojitlán Chinantec) | 16.97% | 2.21% | 52.78% | 9.72% | +14.76% |
| `maa` (Tecóatl Mazatec) | 86.70% | 8.49% | 105.08% | 10.17% | +78.21% |

### Key Findings
- **High Accuracy Gains:** In many tonal languages like Tepehuán (`stp`) and Mazatec (`maa`), the fine-tuning process reduced the error rate from significant levels to nearly zero or double digits.
- **Robustness:** The model shows high resilience against synthetic distortions implemented during the data generation phase.
- **Word-Level Performance:** The relative reduction in Word Error Rate (WER) highlights the model's improved capability in contextualizing character sequences specific to these language families.

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
