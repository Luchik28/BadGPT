import argparse
import sys
import time

import requests

ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "HuggingFaceFW/fineweb-edu"
CONFIG = "sample-10BT"
SPLIT = "train"
PAGE_SIZE = 100


def fetch_subset(max_chars, out_path):
    total_chars = 0
    offset = 0
    docs_written = 0

    with open(out_path, "w", encoding="utf-8") as f:
        while total_chars < max_chars:
            params = {
                "dataset": DATASET,
                "config": CONFIG,
                "split": SPLIT,
                "offset": offset,
                "length": PAGE_SIZE,
            }
            for attempt in range(5):
                try:
                    resp = requests.get(ROWS_URL, params=params, timeout=30)
                    resp.raise_for_status()
                    break
                except requests.RequestException as e:
                    wait = 2 ** attempt
                    print(f"  request failed ({e}), retrying in {wait}s...", file=sys.stderr)
                    time.sleep(wait)
            else:
                raise RuntimeError("failed to fetch rows after 5 attempts")

            data = resp.json()
            rows = data["rows"]
            if not rows:
                print("ran out of rows before hitting max_chars")
                break

            for row in rows:
                text = row["row"]["text"]
                f.write(text)
                f.write("\n\n")
                total_chars += len(text) + 2
                docs_written += 1
                if total_chars >= max_chars:
                    break

            offset += len(rows)
            print(f"  {total_chars:,}/{max_chars:,} chars  ({docs_written} docs, offset {offset})")

    return total_chars, docs_written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-chars", type=int, default=8_000_000,
                         help="stop once this many characters have been written (default: 8,000,000)")
    parser.add_argument("--out", default="fineweb_edu_subset.txt",
                         help="output path (default: fineweb_edu_subset.txt)")
    args = parser.parse_args()

    total_chars, docs_written = fetch_subset(args.max_chars, args.out)
    print(f"done: wrote {total_chars:,} chars from {docs_written} documents to {args.out}")
