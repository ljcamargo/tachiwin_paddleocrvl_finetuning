import os
import json
import tarfile
import random
import shutil
from huggingface_hub import HfApi, login, create_repo

# -----------------------------
# 🔑 Configuration
# -----------------------------
HF_TOKEN = "hf_000000000000000000000000000"  # Replace with your actual token
REPO_ID = "tachiwin/multilingual_ocr_erniesdk"  # New repo for the new format
IMAGES_DIR = "workspace/output/images"
TEXT_DIR = "workspace/output/text"
MIN_IMAGE_SIZE_KB = 20
TRAIN_SPLIT_RATIO = 0.9
INSTRUCTION = "OCR:"

# Sharding configuration
SAMPLES_PER_SHARD = 5000  # Smaller shards for easier tar extraction
TEMP_ASSETS_DIR = "temp_assets"
RESUME_TRAIN_SHARD = 0  
RESUME_TEST_SHARD = 0   

# Local filenames
TRAIN_JSONL = "train.jsonl"
TEST_JSONL = "test.jsonl"

# Login and setup
login(token=HF_TOKEN)
api = HfApi()
create_repo(REPO_ID, token=HF_TOKEN, repo_type="dataset", exist_ok=True)

def find_valid_pairs(images_dir, text_dir):
    """Find all valid image-text pairs and return as list of (basename, image_path, text_path)"""
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
    print(f"✅ Found {len(pairs):,} valid image-text pairs")
    return pairs

def process_and_upload_split(pairs, split_name, start_shard=0, jsonl_path=""):
    print(f"\n📦 Processing {split_name} split ({len(pairs):,} samples)...")
    
    # Open JSONL in append mode
    with open(jsonl_path, "a", encoding="utf-8") as jsonl_f:
        for i in range(start_shard * SAMPLES_PER_SHARD, len(pairs), SAMPLES_PER_SHARD):
            shard_idx = i // SAMPLES_PER_SHARD
            chunk = pairs[i : i + SAMPLES_PER_SHARD]
            
            shard_name = f"{split_name}_shard_{shard_idx:04d}"
            tar_filename = f"{shard_name}.tar.gz"
            local_tar_path = os.path.join(TEMP_ASSETS_DIR, tar_filename)
            
            os.makedirs(TEMP_ASSETS_DIR, exist_ok=True)
            
            print(f"🛠️  Creating {tar_filename}...")
            # Create a Tarball directly containing the images
            with tarfile.open(local_tar_path, "w:gz") as tar:
                for basename, img_path, txt_path in chunk:
                    try:
                        # 1. Add image to tarball (flattened into assets root when expanded)
                        tar.add(img_path, arcname=f"{basename}.png")
                        
                        # 2. Get transcription
                        with open(txt_path, "r", encoding="utf-8") as f:
                            text = ' '.join(f.read().strip().replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ').split())
                        
                        # 3. Create JSONL line in ErnieSDK format
                        sample_json = {
                            "image_info": [
                                # Assuming expanded images will be in ./assets/
                                {"matched_text_index": 0, "image_url": f"./assets/{basename}.png"}
                            ],
                            "text_info": [
                                {"text": INSTRUCTION, "tag": "mask"},
                                {"text": text, "tag": "no_mask"}
                            ]
                        }
                        jsonl_f.write(json.dumps(sample_json, ensure_ascii=False) + "\n")
                        
                    except Exception as e:
                        print(f"❌ Error item {basename}: {e}")

            # 4. Upload Tarball to HF assets/ folder
            print(f"☁️  Uploading {tar_filename} to assets/ folder...")
            api.upload_file(
                path_or_fileobj=local_tar_path,
                path_in_repo=f"assets/{tar_filename}",
                repo_id=REPO_ID,
                repo_type="dataset"
            )
            
            # 5. Cleanup local tarball to save space
            os.remove(local_tar_path)
            print(f"✅ Shard {shard_idx} complete. Total: {i + len(chunk):,}")

def main():
    pairs_list = find_valid_pairs(IMAGES_DIR, TEXT_DIR)
    if not pairs_list: return

    total_samples = len(pairs_list)
    train_split_idx = int(total_samples * TRAIN_SPLIT_RATIO)
    train_pairs = pairs_list[:train_split_idx]
    test_pairs = pairs_list[train_split_idx:]

    # Remove old JSONL files if starting fresh
    if RESUME_TRAIN_SHARD == 0 and os.path.exists(TRAIN_JSONL): os.remove(TRAIN_JSONL)
    if RESUME_TEST_SHARD == 0 and os.path.exists(TEST_JSONL): os.remove(TEST_JSONL)

    # Step 1: Process Train Split
    process_and_upload_split(train_pairs, "train", RESUME_TRAIN_SHARD, TRAIN_JSONL)
    
    # Step 2: Process Test Split
    process_and_upload_split(test_pairs, "test", RESUME_TEST_SHARD, TEST_JSONL)

    # Step 3: Upload final JSONL files
    print("\n📜 Uploading final JSONL manifest files...")
    api.upload_file(path_or_fileobj=TRAIN_JSONL, path_in_repo="train.jsonl", repo_id=REPO_ID, repo_type="dataset")
    api.upload_file(path_or_fileobj=TEST_JSONL, path_in_repo="test.jsonl", repo_id=REPO_ID, repo_type="dataset")

    # Final Cleanup
    if os.path.exists(TEMP_ASSETS_DIR): shutil.rmtree(TEMP_ASSETS_DIR)
    
    print(f"\n🎉 Done! ErnieSDK dataset ready at: https://huggingface.co/datasets/{REPO_ID}")

if __name__ == "__main__":
    main()
