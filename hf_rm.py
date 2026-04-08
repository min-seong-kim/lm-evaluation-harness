from huggingface_hub import HfApi

api = HfApi()

repos = [
    "kmseong/Llama-3.2-3B-instruct-SafeLoRA2",
]

for repo in repos:
    try:
        api.delete_repo(repo_id=repo, repo_type="model")
        print(f"✅ Deleted: {repo}")
    except Exception as e:
        print(f"❌ Failed to delete {repo}: {e}")
