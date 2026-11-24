#!/usr/bin/env python3
"""Render the NIDS system documentation to PDF.

    python docs/documentation/build.py

Requires WeasyPrint (not a runtime dependency of the dashboard):

    pip install weasyprint

The output is docs/documentation/NIDS-System-Documentation.pdf.
"""
from pathlib import Path

from weasyprint import HTML

HERE = Path(__file__).resolve().parent


def main() -> None:
    source = HERE / "src" / "documentation.html"
    target = HERE / "NIDS-System-Documentation.pdf"
    HTML(source).write_pdf(target)
    print(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
