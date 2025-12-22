import os
from PIL import Image
from datasets import Dataset, Features, Image as ImageFeature, Value
from huggingface_hub import HfApi, login, create_repo
import random
import shutil

# -----------------------------
# 🔑 Configuration
# -----------------------------
HF_TOKEN = "hf_000000000000000000000000000"  # Replace with your actual token
REPO_ID = "tachiwin/multilingual_ocr_llm_2"  # Update with your target dataset name
IMAGES_DIR = "workspace/output/images"
TEXT_DIR = "workspace/output/text"
MIN_IMAGE_SIZE_KB = 20
TRAIN_SPLIT_RATIO = 0.9  # 90% train, 10% test
INSTRUCTION = "OCR:"
UPLOAD_LOG_PATH = "workspace/upload_manifest.log"

# Sharding configuration
SAMPLES_PER_SHARD = 2000
TEMP_SHARD_DIR = "temp_shards"
RESUME_TRAIN_SHARD = 12  # Set to the shard index you want to resume from
RESUME_TEST_SHARD = 0   # Set to the shard index you want to resume from

# Login and setup
login(token=HF_TOKEN)
api = HfApi()
create_repo(REPO_ID, token=HF_TOKEN, repo_type="dataset", exist_ok=True)

def find_valid_pairs(images_dir, text_dir):
    """Find all valid image-text pairs and return as list of (basename, image_path, text_path) tuples"""
    pairs = []
    if not os.path.exists(images_dir) or not os.path.exists(text_dir):
        raise FileNotFoundError("Images or Text directory not found")
    
    image_files = {f[:-4]: f for f in os.listdir(images_dir) if f.endswith('.png')}
    text_files = {f[:-4]: f for f in os.listdir(text_dir) if f.endswith('.txt')}
    common_basenames = sorted(list(set(image_files.keys()) & set(text_files.keys())))
    
    min_size_bytes = MIN_IMAGE_SIZE_KB * 1024
    for basename in common_basenames:
        img_path = os.path.join(images_dir, image_files[basename])
        if os.path.getsize(img_path) < min_size_bytes: continue
        txt_path = os.path.join(text_dir, text_files[basename])
        pairs.append((basename, img_path, txt_path))
    
    random.Random(42).shuffle(pairs)
    print(f"✅ Found {len(pairs)} valid image-text pairs")
    return pairs

def convert_sample(basename, img_path, txt_path):
    """Restructure single sample into conversation format"""
    with open(txt_path, "r", encoding="utf-8") as f:
        text = ' '.join(f.read().strip().replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ').split())
    img = Image.open(img_path).convert("RGB")
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text" : INSTRUCTION, "image": None},
            {"type" : "image", "text" : None, "image": img} ]
        },
        { "role" : "assistant", "content" : [{"type" : "text", "text" : text, "image": None}] },
    ]
    return { "images": [img], "messages" : conversation }

def main():
    print(f"🚀 Starting shard-by-shard upload to {REPO_ID}")
    pairs_list = find_valid_pairs(IMAGES_DIR, TEXT_DIR)
    if not pairs_list: return

    total_samples = len(pairs_list)
    train_split_idx = int(total_samples * TRAIN_SPLIT_RATIO)
    train_pairs = pairs_list[:train_split_idx]
    test_pairs = pairs_list[train_split_idx:]

    features = Features({
        "images": [ImageFeature()],
        "messages": [{"content": [{"image": ImageFeature(), "text": Value("string"), "type": Value("string")}], "role": Value("string")}]
    })

    os.makedirs(TEMP_SHARD_DIR, exist_ok=True)

    def process_and_upload_split(pairs, split_name, start_shard=0):
        print(f"\n📦 Processing {split_name} split ({len(pairs)} samples)...")
        if start_shard > 0:
            print(f"⏩ Resuming from shard index {start_shard} (skipping first {start_shard * SAMPLES_PER_SHARD:,} samples)")
            
        for i in range(start_shard * SAMPLES_PER_SHARD, len(pairs), SAMPLES_PER_SHARD):
            shard_idx = i // SAMPLES_PER_SHARD
            chunk = pairs[i : i + SAMPLES_PER_SHARD]
            buffer = []
            processed_samples = 0
            for basename, img_path, txt_path in chunk:
                try:
                    buffer.append(convert_sample(basename, img_path, txt_path))
                    processed_samples += 1
                    print(f"✅ Processed {processed_samples} samples")
                except Exception as e:
                    print(f"❌ Error processing {basename}: {e}")
            
            if buffer:
                # 1. Save locally
                shard_filename = f"{split_name}-{shard_idx:05d}.parquet"
                local_path = os.path.join(TEMP_SHARD_DIR, shard_filename)
                Dataset.from_list(buffer, features=features).to_parquet(local_path)
                
                # 2. Upload to Hub
                print(f"☁️  Uploading shard {shard_idx} ({len(buffer)} samples) | Total processed: {i + len(buffer):,}...")
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=f"data/{shard_filename}",
                    repo_id=REPO_ID,
                    repo_type="dataset"
                )
                
                # 3. Delete locally to save space
                os.remove(local_path)
                
    #process_and_upload_split(train_pairs, "train", RESUME_TRAIN_SHARD)
    process_and_upload_split(test_pairs, "test", RESUME_TEST_SHARD)

    # Cleanup
    if os.path.exists(TEMP_SHARD_DIR):
        shutil.rmtree(TEMP_SHARD_DIR)
    
    print(f"\n🎉 Upload complete! Total pairs: {total_samples}")
    print(f"🔗 Dataset URL: https://huggingface.co/datasets/{REPO_ID}")

if __name__ == "__main__":
    main()