"""
ymliu@2023.5.25
Download a pre-trained model from huggingface.
"""
import os
import argparse
import huggingface_hub


def main(args):
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pretrained_models_dir = os.path.join(project_dir, "pretrained_models")
    os.makedirs(pretrained_models_dir, exist_ok=True)
    repo_id: str = args.repo_id
    repo_name = repo_id.split("/")[-1]
    if repo_id == "HillZhang/pseudo_native_bart_CGEC":
        allow_patterns = [
            "pytorch_model.bin",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "*.md",
            "*.txt",
        ]  # config.json missing "tokenizer_class": "BertTokenizer", in line 77
    else:
        allow_patterns = allow_patterns = ["pytorch_model.bin", "*.json", "*.md", "*.txt"]

    huggingface_hub.snapshot_download(
        repo_id=repo_id,
        local_dir=os.path.join(pretrained_models_dir, repo_name),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", help="A user or an organization name and a repo name separated by a `/`.", required=True)
    args = parser.parse_args()
    main(args)
