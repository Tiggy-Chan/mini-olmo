"""上传模型权重和数据到 Hugging Face Hub。

使用前需要先登录：huggingface-cli login
"""

import argparse
import os
from huggingface_hub import HfApi, create_repo


def upload_checkpoints(repo_id: str, checkpoints_dir: str = "checkpoints"):
    """上传 checkpoint 文件到 HF Hub。"""
    api = HfApi()
    
    for filename in os.listdir(checkpoints_dir):
        if filename.endswith(".pt"):
            local_path = os.path.join(checkpoints_dir, filename)
            print(f"[mini-olmo] 上传 {local_path} ...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=f"checkpoints/{filename}",
                repo_id=repo_id,
            )
            print(f"[mini-olmo] 完成: {filename}")


def upload_tokenizer(repo_id: str, tokenizer_path: str = "tokenizer/tokenizer.json"):
    """上传 tokenizer 到 HF Hub。"""
    if not os.path.exists(tokenizer_path):
        print(f"[mini-olmo] tokenizer 文件不存在: {tokenizer_path}")
        return
    
    api = HfApi()
    print(f"[mini-olmo] 上传 tokenizer ...")
    api.upload_file(
        path_or_fileobj=tokenizer_path,
        path_in_repo="tokenizer/tokenizer.json",
        repo_id=repo_id,
    )
    print("[mini-olmo] tokenizer 上传完成")


def upload_data(repo_id: str, data_dir: str = "data"):
    """上传处理好的数据到 HF Hub（可选）。"""
    api = HfApi()
    
    # 上传 tokenized 数据
    tokenized_dir = os.path.join(data_dir, "tokenized")
    if os.path.exists(tokenized_dir):
        print(f"[mini-olmo] 上传 tokenized 数据 ...")
        api.upload_folder(
            folder_path=tokenized_dir,
            path_in_repo="data/tokenized",
            repo_id=repo_id,
        )
        print("[mini-olmo] tokenized 数据上传完成")


def main():
    parser = argparse.ArgumentParser(description="上传模型到 Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, required=True, help="HF repo ID, 如 username/mini-olmo")
    parser.add_argument("--upload-checkpoints", action="store_true", help="上传 checkpoints")
    parser.add_argument("--upload-tokenizer", action="store_true", help="上传 tokenizer")
    parser.add_argument("--upload-data", action="store_true", help="上传 tokenized 数据")
    parser.add_argument("--all", action="store_true", help="上传所有内容")
    args = parser.parse_args()

    # 创建 repo（如果不存在）
    create_repo(args.repo_id, exist_ok=True)
    print(f"[mini-olmo] Repo: https://huggingface.co/{args.repo_id}")

    if args.all or args.upload_checkpoints:
        upload_checkpoints(args.repo_id)
    
    if args.all or args.upload_tokenizer:
        upload_tokenizer(args.repo_id)
    
    if args.all or args.upload_data:
        upload_data(args.repo_id)

    print("[mini-olmo] 全部上传完成！")


if __name__ == "__main__":
    main()
