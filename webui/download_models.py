"""
Download all Kronos models locally for fast loading.
Run once: python webui/download_models.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import snapshot_download

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

MODELS = [
    ('NeoQuasar/Kronos-Tokenizer-2k', 'Kronos-Tokenizer-2k'),
    ('NeoQuasar/Kronos-Tokenizer-base', 'Kronos-Tokenizer-base'),
    ('NeoQuasar/Kronos-mini', 'Kronos-mini'),
    ('NeoQuasar/Kronos-small', 'Kronos-small'),
    ('NeoQuasar/Kronos-base', 'Kronos-base'),
]


def download_all():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Downloading models to: {MODELS_DIR}\n")

    for repo_id, local_name in MODELS:
        local_path = os.path.join(MODELS_DIR, local_name)
        if os.path.exists(local_path) and os.listdir(local_path):
            print(f"[SKIP] {local_name} — already downloaded")
            continue
        print(f"[DOWNLOADING] {repo_id} -> {local_path}")
        snapshot_download(repo_id=repo_id, local_dir=local_path)
        print(f"[DONE] {local_name}\n")

    print("\nAll models downloaded. They will load instantly from now on.")


if __name__ == '__main__':
    download_all()
