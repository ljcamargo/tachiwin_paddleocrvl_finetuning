import os
from PIL import Image
from datasets import Dataset, DatasetDict, Features, Image as ImageFeature, Value
from huggingface_hub import HfApi, login, create_repo
import random

# -----------------------------
# 🔑 Configuration
# -----------------------------
HF_TOKEN = "hf_000000000000000000000000000"  # Replace with your actual token
REPO_ID = "tachiwin/multilingual_ocr_llm"  # Update with your target dataset name
IMAGES_DIR = "workspace/output/images"
TEXT_DIR = "workspace/output/text"
SHARD_SIZE = 10_000  # Smaller shard size for 30k rows (3 shards total)
MIN_IMAGE_SIZE_KB = 20
TRAIN_SPLIT_RATIO = 0.9  # 90% train, 10% test
COMMIT_MESSAGE = "Update dataset with shuffled shards using datasets library"
UPLOAD_LOG_PATH = "workspace/upload_manifest.log"

# Login and create repo
login(token=HF_TOKEN)
api = HfApi()
create_repo(REPO_ID, token=HF_TOKEN, repo_type="dataset", exist_ok=True)


def find_valid_pairs(images_dir, text_dir):
    """Find all valid image-text pairs and return as list of (basename, image_path, text_path) tuples"""
    pairs = []
    
    # Get all image files
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not os.path.exists(text_dir):
        raise FileNotFoundError(f"Text directory not found: {text_dir}")
    
    image_files = {f[:-4]: f for f in os.listdir(images_dir) if f.endswith('.png')}
    text_files = {f[:-4]: f for f in os.listdir(text_dir) if f.endswith('.txt')}
    
    # Find matching pairs
    common_basenames = set(image_files.keys()) & set(text_files.keys())
    orphan_images = set(image_files.keys()) - common_basenames
    orphan_texts = set(text_files.keys()) - common_basenames
    
    # Report orphans
    if orphan_images:
        print(f"⚠️  Found {len(orphan_images)} orphan images (no matching text)")
    if orphan_texts:
        print(f"⚠️  Found {len(orphan_texts)} orphan texts (no matching image)")
    
    # Create valid pairs list
    min_size_bytes = MIN_IMAGE_SIZE_KB * 1024
    for basename in common_basenames:
        img_path = os.path.join(images_dir, image_files[basename])
        
        # Filter by file size
        if os.path.getsize(img_path) < min_size_bytes:
            continue
            
        txt_path = os.path.join(text_dir, text_files[basename])
        pairs.append((basename, img_path, txt_path))
    
    # Deterministically shuffle to ensure uniform distribution across shards
    random.Random(42).shuffle(pairs)
    
    print(f"✅ Found {len(pairs)} valid image-text pairs")
    return pairs


def prepare_dataset_dict(pairs_list):
    """Prepare data dictionary for datasets library"""
    print(f"📦 Preparing dataset with {len(pairs_list):,} samples...")
    
    data = {
        "id": [],
        "language": [],
        "text": [],
        "image": []
    }
    
    ids_written = []
    
    for basename, img_path, txt_path in pairs_list:
        try:
            # Read text and collapse line breaks
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                # Replace all types of line breaks with space
                text = text.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
                # Optional: Ensure no double spaces after replacement
                text = ' '.join(text.split())
            
            # Add to data dict (datasets library will handle PIL Image automatically)
            data["id"].append(basename)
            data["language"].append(basename[:3])
            data["text"].append(text)
            data["image"].append(img_path)  # Just pass the path, datasets will load it
            
            ids_written.append(basename)
            
        except Exception as e:
            print(f"❌ Error processing {basename}: {e}")
            continue
    
    print(f"✅ Prepared {len(data['id']):,} samples")
    return data, ids_written


def main():
    """Main execution function"""
    print(f"🚀 Starting upload to {REPO_ID}")
    print(f"📊 Using datasets library for proper image handling")
    
    # Find all valid pairs first
    pairs_list = find_valid_pairs(IMAGES_DIR, TEXT_DIR)
    
    if not pairs_list:
        print("❌ No valid image-text pairs found!")
        return
    
    print(f"📝 Total pairs to process: {len(pairs_list):,}")
    
    # Initialize log file
    with open(UPLOAD_LOG_PATH, "w", encoding="utf-8") as log_f:
        log_f.write(f"Upload started\n")
    
    try:
        # Prepare data dictionary
        data_dict, ids_written = prepare_dataset_dict(pairs_list)
        
        # Define features with proper Image type
        features = Features({
            "id": Value("string"),
            "language": Value("string"),
            "text": Value("string"),
            "image": ImageFeature()  # This is the key - proper Image feature type
        })
        
        print("🔨 Creating Dataset object...")
        full_dataset = Dataset.from_dict(data_dict, features=features)
        
        # Split into train and test
        train_size = int(len(full_dataset) * TRAIN_SPLIT_RATIO)
        print(f"📊 Splitting: {train_size:,} train, {len(full_dataset) - train_size:,} test")
        
        dataset_dict = DatasetDict({
            "train": full_dataset.select(range(train_size)),
            "test": full_dataset.select(range(train_size, len(full_dataset)))
        })
        
        # Log all IDs
        with open(UPLOAD_LOG_PATH, "a", encoding="utf-8") as log_f:
            log_f.write(f"\n--- TRAIN SPLIT | Total: {train_size} ---\n")
            log_f.write("\n".join(ids_written[:train_size]) + "\n")
            log_f.write(f"\n--- TEST SPLIT | Total: {len(full_dataset) - train_size} ---\n")
            log_f.write("\n".join(ids_written[train_size:]) + "\n")
        
        print(f"☁️  Uploading to Hugging Face Hub...")
        print(f"   This may take a while for large datasets...")
        
        # Push to hub - datasets library handles everything including sharding
        dataset_dict.push_to_hub(
            REPO_ID,
            token=HF_TOKEN,
            commit_message=COMMIT_MESSAGE,
            max_shard_size=f"{SHARD_SIZE}MB"  # Control shard size
        )
        
        print(f"\n🎉 Upload complete! Total samples: {len(full_dataset):,}")
        print(f"   Train: {len(dataset_dict['train']):,}")
        print(f"   Test: {len(dataset_dict['test']):,}")
        print(f"🔗 Dataset URL: https://huggingface.co/datasets/{REPO_ID}")
        print(f"📜 Manifest saved to: {UPLOAD_LOG_PATH}")
        
        # Verify the upload
        print("\n🔍 Verifying dataset can be loaded...")
        from datasets import load_dataset
        test_load = load_dataset(REPO_ID, split="train", streaming=True)
        first_example = next(iter(test_load))
        print(f"✅ Verification successful! First example keys: {list(first_example.keys())}")
        print(f"   Image type: {type(first_example['image'])}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Upload interrupted by user")
    except Exception as e:
        error_msg = f"❌ Error during upload: {e}"
        print(error_msg)
        with open(UPLOAD_LOG_PATH, "a", encoding="utf-8") as log_f:
            log_f.write(f"\nCRITICAL ERROR: {error_msg}\n")
        raise


if __name__ == "__main__":
    main()