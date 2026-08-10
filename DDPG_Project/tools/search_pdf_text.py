from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("keywords", nargs="+")
    args = parser.parse_args()

    reader = PdfReader(args.path)
    print(f"PAGES: {len(reader.pages)}")
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        matched = [keyword for keyword in args.keywords if keyword.lower() in text.lower()]
        if matched:
            print(f"\n=== PAGE {page_number} | {', '.join(matched)} ===")
            print(text)


if __name__ == "__main__":
    main()
