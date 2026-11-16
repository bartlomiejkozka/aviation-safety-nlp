#!/usr/bin/env python3
"""Merge yearly ASRS CSV exports (data/<year>.csv) into a single CSV.

ASRS exports have two header rows (category row + column-name row), so the
merged file keeps that same layout and stays readable with
``pd.read_csv(path, header=1)``.

- Files are processed in filename order (2019.csv, 2020.csv, ...).
- Both header rows must match across files; otherwise the script stops.
- Blank rows are dropped.
- Reports appearing in more than one file (same ACN) are kept once.

Usage:
    python merge_csv.py                      # data/*.csv -> data/merged.csv
    python merge_csv.py -i data -o out.csv
    python merge_csv.py --keep-duplicates
"""

import argparse
import csv
import sys
from pathlib import Path

HEADER_ROWS = 2
ID_COLUMN = "ACN"


def read_rows(path: Path) -> list[list[str]]:
    for encoding in ("utf-8-sig", "latin1"):
        try:
            with path.open(encoding=encoding, newline="") as f:
                return list(csv.reader(f))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode {path}")


def merge(files: list[Path], output: Path, dedupe: bool) -> None:
    header: list[list[str]] | None = None
    id_idx: int | None = None
    seen: set[str] = set()
    merged: list[list[str]] = []
    duplicates = 0

    for path in files:
        rows = read_rows(path)
        file_header, body = rows[:HEADER_ROWS], rows[HEADER_ROWS:]

        if header is None:
            header = file_header
            if dedupe:
                if ID_COLUMN not in header[-1]:
                    sys.exit(f"Column '{ID_COLUMN}' not found in {path}; use --keep-duplicates")
                id_idx = header[-1].index(ID_COLUMN)
        elif file_header != header:
            sys.exit(f"Header of {path} differs from {files[0]}; aborting.")

        added = 0
        for row in body:
            if not any(cell.strip() for cell in row):
                continue
            if id_idx is not None:
                key = row[id_idx]
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
            merged.append(row)
            added += 1
        print(f"{path.name}: {added} rows")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(header)
        writer.writerows(merged)

    print(f"\nWrote {len(merged)} rows from {len(files)} files to {output}")
    if duplicates:
        print(f"Skipped {duplicates} duplicate rows (same {ID_COLUMN})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-i", "--input-dir", type=Path, default=Path("data"))
    parser.add_argument("-o", "--output", type=Path, default=Path("data/merged.csv"))
    parser.add_argument("--keep-duplicates", action="store_true",
                        help=f"don't drop rows with a repeated {ID_COLUMN}")
    args = parser.parse_args()

    output = args.output.resolve()
    files = sorted(p for p in args.input_dir.glob("*.csv") if p.resolve() != output)
    if not files:
        sys.exit(f"No CSV files found in {args.input_dir}")

    merge(files, args.output, dedupe=not args.keep_duplicates)


if __name__ == "__main__":
    main()
