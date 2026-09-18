# ingest_corpus.py
"""GitHub Repository and Code Corpus Ingestion Utility for Parhi-GPT.

Allows ingesting downloaded GitHub repositories, local folders of code,
markdown docs, and text files into ``parhi_corpus.txt`` for fine-tuning
the local Parhi Transformer model with ``python train.py``.

Usage:
    # Ingest a downloaded or cloned GitHub repository folder:
    python ingest_corpus.py --dir /path/to/cloned/repo

    # Ingest a specific code or documentation file:
    python ingest_corpus.py --file /path/to/file.py

    # Clone a public GitHub repository directly and ingest it:
    python ingest_corpus.py --git https://github.com/user/repository.git

    # Dry run to inspect files and statistics without modifying corpus:
    python ingest_corpus.py --dir /path/to/repo --dry-run
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile


# Supported text and code extensions
VALID_EXTENSIONS = {
    ".py", ".md", ".txt", ".js", ".ts", ".jsx", ".tsx",
    ".html", ".css", ".json", ".c", ".cpp", ".h", ".hpp",
    ".java", ".go", ".rs", ".rb", ".php", ".sh", ".yaml", ".yml",
}

# Directories to ignore
IGNORE_DIRS = {
    ".git", ".github", "node_modules", "__pycache__", ".pytest_cache",
    ".venv", "venv", "env", "dist", "build", ".idea", ".vscode",
    "bin", "obj", "vendor", "coverage", "site-packages",
}

# Files to ignore
IGNORE_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "Cargo.lock", "go.sum",
}


def is_text_file(filepath: str) -> bool:
    """Check if a file has a valid extension and does not contain binary null bytes."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in VALID_EXTENSIONS:
        return False
    if os.path.basename(filepath) in IGNORE_FILES:
        return False
    # Check for minified files
    if filepath.endswith(".min.js") or filepath.endswith(".min.css"):
        return False

    try:
        with open(filepath, "rb") as f:
            chunk = f.read(1024)
            if b"\0" in chunk:
                return False
        return True
    except Exception:
        return False


def format_code_as_dialogue(filepath: str, content: str) -> str:
    """Format source code or documentation into a User/Parhi dialogue turn."""
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower()

    # Clean whitespace and strip excessive blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", content.strip())
    if not cleaned or len(cleaned) < 30:
        return ""

    # Truncate single files if too massive (keep first 2500 chars)
    if len(cleaned) > 2500:
        cleaned = cleaned[:2500] + "\n... [truncated for training context]"

    if ext == ".md":
        return f"User: Can you explain the documentation from {filename}?\nParhi: Here is the documentation:\n\n{cleaned}\n\n"
    else:
        return f"User: How is {filename} structured and what does it implement?\nParhi: Here is the source code implementation for {filename}:\n\n```\n{cleaned}\n```\n\n"


def process_directory(directory: str) -> list[tuple[str, str]]:
    """Scan directory and return list of (filepath, formatted_dialogue)."""
    results: list[tuple[str, str]] = []

    for root, dirs, files in os.walk(directory):
        # Prune ignored directories in-place
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]

        for file in sorted(files):
            full_path = os.path.join(root, file)
            if is_text_file(full_path):
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    dialogue = format_code_as_dialogue(full_path, text)
                    if dialogue:
                        results.append((full_path, dialogue))
                except Exception as e:
                    print(f"  [!] Skipped {file}: {e}")

    return results


def clone_github_repo(repo_url: str, dest_dir: str) -> bool:
    """Clone a GitHub repository via git CLI."""
    print(f"[*] Cloning repository from {repo_url}...")
    try:
        cmd = ["git", "clone", "--depth", "1", repo_url, dest_dir]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[!] Error running git clone: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest GitHub repositories or code folders into Parhi training corpus."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", help="Path to local folder or cloned GitHub repository.")
    group.add_argument("--file", help="Path to an individual code or text file.")
    group.add_argument("--git", help="GitHub clone URL (e.g. https://github.com/user/repo.git).")

    parser.add_argument(
        "--corpus",
        default="parhi_corpus.txt",
        help="Target corpus file to append to (default: parhi_corpus.txt)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and print statistics without writing to corpus file.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=200,
        help="Maximum number of files to ingest from a repository (default: 200)",
    )

    args = parser.parse_args()

    target_dir = ""
    temp_dir = None

    if args.git:
        temp_dir = tempfile.mkdtemp(prefix="parhi_repo_")
        ok = clone_github_repo(args.git, temp_dir)
        if not ok:
            shutil.rmtree(temp_dir, ignore_errors=True)
            sys.exit(1)
        target_dir = temp_dir
    elif args.dir:
        if not os.path.isdir(args.dir):
            print(f"[!] Directory not found: {args.dir}")
            sys.exit(1)
        target_dir = args.dir

    ingested_dialogues: list[tuple[str, str]] = []

    if args.file:
        if not os.path.isfile(args.file):
            print(f"[!] File not found: {args.file}")
            sys.exit(1)
        if is_text_file(args.file):
            with open(args.file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            d = format_code_as_dialogue(args.file, content)
            if d:
                ingested_dialogues.append((args.file, d))
    else:
        print(f"[*] Scanning files in '{target_dir}'...")
        ingested_dialogues = process_directory(target_dir)

    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)

    if not ingested_dialogues:
        print("[!] No eligible text or code files found to ingest.")
        sys.exit(0)

    # Limit to max_files to maintain balanced corpus
    if len(ingested_dialogues) > args.max_files:
        print(f"[*] Capping ingestion to {args.max_files} files (out of {len(ingested_dialogues)} found).")
        ingested_dialogues = ingested_dialogues[:args.max_files]

    total_chars = sum(len(d) for _, d in ingested_dialogues)
    est_tokens = total_chars // 4

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"Files Processed  : {len(ingested_dialogues)}")
    print(f"Total Characters : {total_chars:,}")
    print(f"Estimated Tokens : ~{est_tokens:,}")
    print(f"Target Corpus    : {args.corpus}")
    print("=" * 60)

    if args.dry_run:
        print("[*] Dry run complete. No files were modified.")
        return

    # Append to corpus file
    corpus_path = os.path.abspath(args.corpus)
    with open(corpus_path, "a", encoding="utf-8") as f:
        f.write("\n\n# --- Ingested Repository Knowledge ---\n\n")
        for filepath, dialogue in ingested_dialogues:
            f.write(dialogue)

    print(f"[✓] Successfully appended {len(ingested_dialogues)} entries into '{args.corpus}'.")
    print("[✓] Ready to train! Run the following command to update your model weights:")
    print("\n    python train.py\n")


if __name__ == "__main__":
    main()
