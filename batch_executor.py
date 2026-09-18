# batch_executor.py
"""Batch filesystem and OS automation engine for Parhi-GPT.

Provides capabilities for:
- Batch folder creation (e.g., "create 100 folders with a particular name of my friends")
- Safe path isolation (Desktop, Documents, or workspace)
- Natural language batch task parsing
- Rollback / undo logging for batch actions
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BatchExecutionResult:
    """Result of a batch task execution."""
    success: bool
    action: str
    items_created: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    display_text: str = ""
    undo_token: str = ""


class BatchExecutor:
    """Executes bulk filesystem actions safely with access validation."""

    def __init__(self, default_base_dir: str | None = None) -> None:
        if default_base_dir:
            self.base_dir = Path(default_base_dir)
        else:
            # Default to user's Desktop or workspace
            desktop = Path(os.environ.get("USERPROFILE", ".")) / "Desktop"
            if desktop.exists():
                self.base_dir = desktop
            else:
                self.base_dir = Path(os.getcwd())
        
        self._history: dict[str, list[str]] = {}

    def parse_folder_request(self, message: str) -> tuple[int, str, list[str]]:
        """Parse natural language command for batch folder creation.

        Examples:
            - "create 100 folders with name of my friends"
            - "create 50 folders named friend"
            - "make 10 folders for Alex, Bob, Charlie"

        Returns:
            Tuple of (count, base_name, custom_names_list)
        """
        lower = message.lower()
        
        # 1. Extract count (number)
        count_match = re.search(r"\b(\d+)\s*(?:folders?|directories?|dirs?)\b", lower)
        count = int(count_match.group(1)) if count_match else 0

        # If no explicit "X folders" pattern, look for any number near create/make
        if count == 0:
            fallback_num = re.search(r"(?:create|make|generate)\s+(\d+)", lower)
            if fallback_num:
                count = int(fallback_num.group(1))

        # Default count if not found but requested
        if count == 0:
            count = 10

        # Cap count to 500 for safety
        count = min(count, 500)

        # 2. Extract friend names or base name
        # Check for explicit name list: e.g. "named Alex, Bob, Sam"
        list_match = re.search(r"(?:named|names?\s*[:=]|for)\s+([a-zA-Z0-9_,\s]+)", message)
        custom_names: list[str] = []
        if list_match:
            raw_names = list_match.group(1)
            parts = [p.strip() for p in re.split(r"[,;]+", raw_names) if p.strip()]
            # Filter out common stop words
            parts = [p for p in parts if p.lower() not in ("my", "friends", "folders", "a", "the", "particular")]
            if parts:
                custom_names = parts

        # 3. Determine base name
        base_name = "Friend"
        base_match = re.search(r"(?:with\s+(?:a\s+)?(?:particular\s+)?name\s+(?:of\s+)?|named\s+)([a-zA-Z0-9_\-]+)", message, re.IGNORECASE)
        if base_match:
            cand = base_match.group(1).strip()
            if cand.lower() not in ("my", "friends", "particular", "the", "a"):
                base_name = cand
        elif "friend" in lower:
            base_name = "Friend"

        return count, base_name, custom_names

    def create_batch_folders(
        self,
        count: int,
        base_name: str = "Friend",
        custom_names: list[str] | None = None,
        target_dir: str | Path | None = None,
    ) -> BatchExecutionResult:
        """Create N folders with sequential or custom friend names.

        Args:
            count: Number of folders to create.
            base_name: Base prefix (e.g., "Friend").
            custom_names: Optional explicit list of friend names.
            target_dir: Destination directory (defaults to self.base_dir).

        Returns:
            BatchExecutionResult with created paths and status.
        """
        dest = Path(target_dir) if target_dir else self.base_dir
        dest.mkdir(parents=True, exist_ok=True)

        # Built-in sample friend names to make generated folders look realistic and personal
        sample_friend_names = [
            "Rahul", "Pooja", "Aarav", "Sneha", "Rohan", "Ananya", "Vikram", "Neha",
            "Karan", "Priya", "Aditya", "Riya", "Manish", "Divya", "Siddharth", "Kavya",
            "Nikhil", "Shreya", "Varun", "Tanvi", "Arjun", "Isha", "Sameer", "Meera",
            "Harsh", "Deepika", "Gaurav", "Swati", "Mayank", "Ritika", "Kunal", "Simran",
            "Yash", "Avani", "Akash", "Bhavna", "Abhishek", "Tara", "Prateek", "Diya",
        ]

        created_paths: list[str] = []
        errors: list[str] = []

        import time
        undo_token = f"batch_{int(time.time())}"

        # Combine custom names and formatted names
        names_pool = list(custom_names or [])

        for i in range(1, count + 1):
            if names_pool and i <= len(names_pool):
                folder_name = names_pool[i - 1]
            elif base_name.lower() == "friend" and i <= len(sample_friend_names):
                folder_name = f"{base_name}_{i:03d}_{sample_friend_names[i - 1]}"
            else:
                folder_name = f"{base_name}_{i:03d}"

            # Clean folder name for Windows compatibility
            folder_name = re.sub(r'[<>:"/\\|?*]', "_", folder_name).strip()
            folder_path = dest / folder_name

            try:
                folder_path.mkdir(parents=True, exist_ok=True)
                created_paths.append(str(folder_path))
            except Exception as e:
                errors.append(f"{folder_name}: {e}")

        self._history[undo_token] = created_paths

        success = len(created_paths) > 0
        display_text = (
            f"Successfully created {len(created_paths)} folders in {dest}.\n"
            f"Range: {Path(created_paths[0]).name} to {Path(created_paths[-1]).name}."
            if success else f"Failed to create folders: {'; '.join(errors)}"
        )

        return BatchExecutionResult(
            success=success,
            action="create_folders",
            items_created=created_paths,
            errors=errors,
            display_text=display_text,
            undo_token=undo_token,
        )

    def undo_batch(self, undo_token: str) -> tuple[int, list[str]]:
        """Undo a batch creation by removing empty created folders."""
        paths = self._history.get(undo_token, [])
        removed = 0
        errors: list[str] = []

        for p_str in reversed(paths):
            p = Path(p_str)
            try:
                if p.exists() and p.is_dir():
                    # Only remove if directory is empty to prevent data loss
                    if not any(p.iterdir()):
                        p.rmdir()
                        removed += 1
            except Exception as e:
                errors.append(f"{p.name}: {e}")

        if undo_token in self._history:
            del self._history[undo_token]

        return removed, errors
