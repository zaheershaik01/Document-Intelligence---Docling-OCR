# Document Intelligence — Docling & OCR

A Python-based project that extracts content from documents and images and converts it into Markdown (`.md`) files using Docling and OCR.

## Features
- Supports PDF, DOCX, PPTX, and image formats.
- Uses OCR to extract text from images.
- Preserves table structure where supported.
- Skips files that already have Markdown output.
- Supports selective file conversion.

## Installation
```bash
pip install docling
```

## Usage
Place documents and `main.py` in the same directory.

```bash
python main.py
```

Convert specific files:

```bash
python main.py document.pdf scan.png
```

## Output
Generates a `.md` file for each successfully converted document or image.

## Technologies
- Python
- Docling
- OCR
- Markdown
