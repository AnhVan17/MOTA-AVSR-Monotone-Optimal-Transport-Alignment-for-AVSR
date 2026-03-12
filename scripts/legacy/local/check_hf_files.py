
from huggingface_hub import list_repo_files

repo_id = "nguyenvulebinh/ViCocktail"
try:
    files = list_repo_files(repo_id, repo_type="dataset")
    print(f"Files in {repo_id} (Filtering for 'test'):")
    for f in files:
        if "test" in f:
            print(f" - {f}")
            
    print("\nFiles in {repo_id} (Filtering for 'train' first 5):")
    count = 0
    for f in files:
        if "train" in f and count < 5:
            print(f" - {f}")
            count += 1
            
except Exception as e:
    print(f"Failed: {e}")
