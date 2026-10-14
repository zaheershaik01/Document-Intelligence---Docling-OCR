import sys
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
    PowerpointFormatOption,
    WordFormatOption,
)


BASE_DIR = Path(__file__).resolve().parent
REQUESTED_FILES = {Path(argument).name for argument in sys.argv[1:]}
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}
INPUT_FILES = sorted(
    (
        file_path
        for file_path in BASE_DIR.iterdir()
        if file_path.is_file()
        and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not file_path.with_suffix(".md").exists()
        and file_path.name != Path(__file__).name
        and (not REQUESTED_FILES or file_path.name in REQUESTED_FILES)
    ),
    key=lambda file_path: (file_path.suffix.lower() == ".pdf", file_path.name),
)

if not INPUT_FILES:
    print("All supported files already have Markdown files.")
    raise SystemExit(0)

pdf_pipeline_options = PdfPipelineOptions(
    do_ocr=False,
    do_table_structure=True,
    generate_page_images=False,
    generate_picture_images=False,
)
image_pipeline_options = PdfPipelineOptions(
    do_ocr=True,
    do_table_structure=True,
    generate_page_images=False,
    generate_picture_images=False,
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pdf_pipeline_options
        ),
        InputFormat.IMAGE: ImageFormatOption(
            pipeline_options=image_pipeline_options
        ),
        InputFormat.DOCX: WordFormatOption(),
        InputFormat.PPTX: PowerpointFormatOption(),
    }
)

for input_path in INPUT_FILES:
    print(f"Converting: {input_path.name}", flush=True)
    markdown_path = input_path.with_suffix(".md")
    try:
        result = converter.convert(input_path)
        markdown_path.write_text(
            result.document.export_to_markdown(), encoding="utf-8"
        )
        print(f"Created: {markdown_path.name}", flush=True)
    except Exception as error:
        print(f"Failed: {input_path.name}: {error}", flush=True)

