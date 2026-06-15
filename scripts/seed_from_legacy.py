from __future__ import annotations

import argparse
from pathlib import Path

from wonderbot.agent import WonderBot
from wonderbot.config import WonderBotConfig


TEXT_SUFFIXES = {".md", ".txt", ".py", ".rs", ".toml"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seed WonderBot memory from a legacy corpus tree.")
    parser.add_argument("legacy_root", help="Path to old project tree.")
    parser.add_argument("--config", default="configs/default.toml", help="WonderBot config path.")
    parser.add_argument("--max-bytes", type=int, default=200_000, help="Skip files larger than this many bytes.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = WonderBotConfig.load(args.config)
    bot = WonderBot(cfg)

    root = Path(args.legacy_root)
    if not root.exists():
        raise SystemExit(f"Legacy root does not exist: {root}")

    count = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.stat().st_size > args.max_bytes:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        text = text.strip()
        if not text:
            continue
        header = f"[legacy:{path.relative_to(root)}]"
        bot.memory.add(f"{header}\n{text}", source="legacy", metadata={"path": str(path.relative_to(root))})
        count += 1

    bot.save()
    print(f"Seeded {count} legacy files into {cfg.memory.path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
