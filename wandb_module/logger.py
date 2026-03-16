from __future__ import annotations

from pathlib import Path
from typing import Any


def _to_serializable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _to_serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_serializable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


class WandbLogger:
    """
    Rank-0 only W&B logger with no-op fallback.
    Config example:
      wandb:
        enabled: false
        project: "calib-net"
        entity: null
        run_name: null
        tags: []
        notes: ""
        mode: "online"  # online/offline/disabled
    """

    def __init__(self, cfg: dict, out_dir: Path, rank: int):
        wb = cfg.get("wandb", {})
        self.enabled = bool(wb.get("enabled", False)) and rank == 0
        self.run = None
        self._wandb = None

        if not self.enabled:
            return

        try:
            import wandb  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "wandb is enabled in config, but import failed. Install it with `pip install wandb`."
            ) from e

        self._wandb = wandb
        out_dir.mkdir(parents=True, exist_ok=True)

        run_name = wb.get("run_name") or cfg.get("exp", {}).get("name", None)
        self.run = wandb.init(
            project=wb.get("project", "calib-net"),
            entity=wb.get("entity", None),
            name=run_name,
            tags=list(wb.get("tags", [])),
            notes=wb.get("notes", ""),
            mode=wb.get("mode", "online"),
            dir=str(out_dir),
            config=_to_serializable(cfg),
        )

    @property
    def is_active(self) -> bool:
        return self.run is not None and self._wandb is not None

    def watch_model(self, model, log: str = "gradients", log_freq: int = 500):
        if not self.is_active:
            return
        self._wandb.watch(model, log=log, log_freq=int(log_freq))

    def log(self, data: dict[str, Any], step: int | None = None):
        if not self.is_active:
            return
        payload = _to_serializable(data)
        if step is None:
            self._wandb.log(payload)
        else:
            self._wandb.log(payload, step=int(step))

    def finish(self):
        if not self.is_active:
            return
        self._wandb.finish()
