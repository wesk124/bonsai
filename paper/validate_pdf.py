#!/usr/bin/env python3
"""Check the rendered manuscript's content and embedded result figures."""
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from pypdf import PdfReader


def main():
    directory = Path(__file__).resolve().parent
    pdf = directory / "manuscript.pdf"
    reader = PdfReader(pdf)
    text = unicodedata.normalize("NFKC", "\n".join(p.extract_text() or "" for p in reader.pages))
    compact = re.sub(r"\s+", "", text)
    expected = ["JieChen", "ShengWei", "RULER", "INT8", "INT4", "Triton", "900", "6,300",
                "378", "189", "216", "49.03", "43.62", "79.76", "124.11", "144.28",
                "2.95", "5.72", "inconclusive", "References"]
    missing = [s for s in expected if s not in compact]
    if missing:
        raise ValueError(f"Rendered manuscript is missing required content: {missing}")
    if re.search(r"\?\?|\{\{PAIR_|PLACEHOLDER", text):
        raise ValueError("Unresolved reference or manuscript placeholder in rendered PDF")
    log = (directory / "manuscript.log").read_text(errors="replace")
    unresolved = [line for line in log.splitlines()
                  if re.search(r"(?:Citation|Reference).*undefined|There were undefined|^!", line)]
    if unresolved:
        raise ValueError(f"Unresolved LaTeX errors or citations: {unresolved}")
    image_counts = []
    for page in reader.pages:
        resources = page.get("/Resources", {}).get_object()
        objects = resources.get("/XObject", {}).get_object() if "/XObject" in resources else {}
        image_counts.append(sum(obj.get_object().get("/Subtype") == "/Image" for obj in objects.values()))
    figures = sorted((directory / "figures").glob("*.png"))
    if len(figures) < 5 or sum(image_counts) < len(figures):
        raise ValueError("Expected multiple result figures embedded in the manuscript")
    metadata = reader.metadata
    if not metadata or "Jie Chen" not in str(metadata.author) or "Sheng Wei" not in str(metadata.author):
        raise ValueError("Both manuscript authors must be recorded in PDF metadata")
    warnings = [line for line in log.splitlines() if "Overfull" in line or "LaTeX Warning:" in line]
    result = {"status": "PASS", "pages": len(reader.pages), "title": str(metadata.title),
              "authors": str(metadata.author), "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
              "figure_png_files": [p.name for p in figures], "embedded_images_per_page": image_counts,
              "required_content_checks": len(expected), "unresolved_references": 0,
              "layout_warnings_for_visual_review": warnings,
              "visual_review": "Required separately; page rendering and inspection are not replaced by these checks."}
    destination = directory / "validation.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
