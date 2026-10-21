"""Process-wide settings that code deep inside the dataset, importer and managers consults.

The desktop CLI and tests use the defaults below; `yolo-workbench-server` overwrites them from its
configuration at startup (`configure`). Kept in one module-level object rather than threaded through
every constructor, because they are limits and switches, not state.
"""

from __future__ import annotations

from dataclasses import dataclass

MB = 1024 * 1024


@dataclass
class Runtime:
    server_mode: bool = False
    allow_path_import: bool = True  # import from a folder named by path (off on a server: browser uploads only)
    scan_cwd_for_models: bool = True  # model discovery also lists weights in the working directory
    max_upload_file_bytes: int = 8 * 1024 * MB
    max_upload_files: int = 0  # per upload (staging dir); 0 = unlimited
    max_upload_total_bytes: int = 0  # per upload; 0 = unlimited
    max_model_upload_bytes: int = 16 * 1024 * MB
    min_free_bytes: int = 0  # refuse uploads, imports, runs and exports below this much free space; 0 = no check
    undo_max_transactions: int = 500
    undo_max_bytes: int = 512 * MB
    gpu_models_during_training: str = "allow"  # or "refuse": no SAM/prediction model loads while a run trains
    embeddings_enabled: bool = True
    trash_retention_days: float = 0  # dataset trash of earlier sessions deleted after this many days; 0 = keep
    backup_retention_days: float = 0


RUNTIME = Runtime()


def configure(config) -> Runtime:
    """Apply a ServerConfig to the process-wide settings."""
    RUNTIME.server_mode = True
    RUNTIME.allow_path_import = False
    RUNTIME.scan_cwd_for_models = False
    RUNTIME.max_upload_file_bytes = config.max_upload_file_mb * MB
    RUNTIME.max_upload_files = config.max_upload_files
    RUNTIME.max_upload_total_bytes = config.max_upload_total_mb * MB
    RUNTIME.max_model_upload_bytes = config.max_model_upload_mb * MB
    RUNTIME.min_free_bytes = int(config.min_free_disk_gb * 1024 * MB)
    RUNTIME.undo_max_transactions = config.undo_max_transactions
    RUNTIME.undo_max_bytes = config.undo_max_mb * MB
    RUNTIME.gpu_models_during_training = config.gpu_models_during_training
    RUNTIME.embeddings_enabled = config.embeddings_enabled
    RUNTIME.trash_retention_days = config.trash_retention_days
    RUNTIME.backup_retention_days = config.backup_retention_days
    return RUNTIME
