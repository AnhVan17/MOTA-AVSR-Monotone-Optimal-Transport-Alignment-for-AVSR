
import os
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

REPO_ID = "nguyenvulebinh/ViCocktail"


def download_single_shard(output_dir: str, shard: str):
    """Download ONE specific .tar shard (small, for local testing).

    shard: filename like 'avvn-train-000098.tar' (smallest is ~207MB).
    """
    filename = shard if shard.startswith("data/") else f"data/{shard}"
    logger.info(f"Downloading single shard {filename} from {REPO_ID}...")
    path = hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename=filename, local_dir=output_dir
    )
    logger.info(f"Saved: {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return path

def download_vicocktail(output_dir: str, subsets: list = None):
    """
    Download ViCocktail dataset shards from HuggingFace.
    
    Args:
        output_dir: Local path to save data
        subsets: List of subsets to download (e.g., ['train', 'test_snr_0_interferer_1'])
                 If None, downloads everything (Huge!).
    """
    repo_id = "nguyenvulebinh/ViCocktail"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading from {repo_id} to {output_path}...")
    
    # Default patterns if not specified
    if not subsets:
        logger.warning("No subsets specified. Downloading ALL training data (this might take a while).")
        allow_patterns = [
            "data/avvn-train-*.tar",  # All training shards (Corrected prefix)
            "data/avvn-test_*.tar",   # All test shards (Corrected prefix)
        ]
    else:
        allow_patterns = []
        for subset in subsets:
            if subset == 'train':
                allow_patterns.append("data/avvn-train-*.tar")
            elif 'test' in subset:
                # E.g., data/avvn-test_snr_0_interferer_1-*.tar
                allow_patterns.append(f"data/avvn-{subset}-*.tar")
    
    logger.info(f"Targeting patterns: {allow_patterns}")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=output_dir,
            allow_patterns=allow_patterns,
            max_workers=8
        )
        logger.info("Download completed successfully.")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ViCocktail Dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the dataset")
    parser.add_argument("--subsets", nargs="+", default=['train'],
                        help="Subsets to download (default: train). Pass 'all' for everything. (LARGE)")
    parser.add_argument("--shard", type=str, default=None,
                        help="Download ONE shard only, e.g. 'avvn-train-000098.tar' (small, for local test)")

    args = parser.parse_args()

    # Single-shard mode (recommended for local testing — avoids huge multi-GB downloads).
    if args.shard:
        download_single_shard(args.output_dir, args.shard)
    else:
        subsets = args.subsets
        if 'all' in subsets:
            subsets = None
        download_vicocktail(args.output_dir, subsets)
