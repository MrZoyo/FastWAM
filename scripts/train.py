import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = str(REPO_ROOT / "src")
if SRC_ROOT in sys.path:
    sys.path.remove(SRC_ROOT)
sys.path.insert(0, SRC_ROOT)

import hydra
from omegaconf import DictConfig

import fastwam
from fastwam.runtime import run_training
from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    print(
        f"[source] rank={rank} cwd={Path.cwd()} python={sys.executable} "
        f"fastwam={Path(fastwam.__file__).resolve()}",
        flush=True,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
