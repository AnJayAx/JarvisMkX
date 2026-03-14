"""
PDF Processing Module for Jarvis Mk.X
Extracts and chunks text from research paper PDFs.

Usage:
    from processor import PaperProcessor
    processor = PaperProcessor(chunk_size=512, chunk_overlap=50)
    paper = processor.process("path/to/paper.pdf")
"""

import fitz  # PyMuPDF
import re
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class TextChunk:
    """A chunk of text from a research paper with metadata."""
    chunk_id: int
    text: str
    section: str
    page_numbers: List[int]
    chunk_type: str = "text"
    start_char: int = 0
    token_count_approx: int = 0


@dataclass
class ProcessedPaper:
    """A fully processed research paper."""
    title: str
    authors: str
    abstract: str
    full_text: str
    sections: Dict[str, str]
    chunks: List[TextChunk] = field(default_factory=list)
    num_pages: int = 0
    metadata: Dict = field(default_factory=dict)


class PaperProcessor:
    """
    Processes a research paper PDF into structured, chunked text.

    Pipeline:
        PDF -> Raw Extraction -> Section Detection -> Chunking -> ProcessedPaper
    """

    def __init__(self, chunk_size=512, chunk_overlap=50, chars_per_token=4):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chars_per_token = chars_per_token
        self.chunk_size_chars = chunk_size * chars_per_token
        self.chunk_overlap_chars = chunk_overlap * chars_per_token

    # ─── Main Entry Point ───

    def process(self, pdf_path: str) -> ProcessedPaper:
        """Process a PDF file into a structured ProcessedPaper."""
        doc = fitz.open(pdf_path)
        try:
            pages_data = self._extract_pages(doc)
            sections, full_text = self._detect_sections(pages_data)
            title, authors = self._extract_metadata(pages_data, doc)
            abstract = self._extract_abstract(sections, full_text)
            chunks = self._chunk_text(sections, pages_data)
            return ProcessedPaper(
                title=title,
                authors=authors,
                abstract=abstract,
                full_text=full_text,
                sections=sections,
                chunks=chunks,
                num_pages=len(doc),
                metadata=dict(doc.metadata) if doc.metadata else {},
            )
        finally:
            doc.close()

    # ─── Step 1: Extract Pages ───

    def _extract_pages(self, doc) -> List[Dict]:
        """Extract text from each page with font size information."""
        pages_data = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            page_info = {
                "page_num": page_num + 1,
                "text_blocks": [],
                "raw_text": page.get_text(),
            }
            for block in blocks:
                if block["type"] != 0:  # Skip image blocks
                    continue
                block_text = ""
                block_fonts = []
                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span["text"]
                        block_fonts.append({
                            "size": span["size"],
                            "flags": span["flags"],
                            "font": span["font"],
                        })
                    block_text += line_text + "\n"
                block_text = block_text.strip()
                if not block_text:
                    continue
                if block_fonts:
                    sizes = [f["size"] for f in block_fonts]
                    dominant_size = max(set(sizes), key=sizes.count)
                    is_bold = any(f["flags"] & 16 for f in block_fonts)
                else:
                    dominant_size = 10.0
                    is_bold = False
                page_info["text_blocks"].append({
                    "text": block_text,
                    "font_size": round(dominant_size, 1),
                    "is_bold": is_bold,
                    "bbox": block["bbox"],
                })
            pages_data.append(page_info)
        return pages_data

    # ─── Step 2: Detect Sections ───

    def _detect_sections(self, pages_data: List[Dict]) -> tuple:
        """
        Detect section headers based on font size and formatting.
        Returns (sections_dict, full_text_string).
        """
        # First pass: find body text font size
        all_sizes = []
        for page in pages_data:
            for block in page["text_blocks"]:
                text_len = len(block["text"])
                all_sizes.extend([block["font_size"]] * text_len)

        if not all_sizes:
            return {"Full Text": ""}, ""

        body_font_size = max(set(all_sizes), key=all_sizes.count)

        # Section header patterns
        section_pattern = re.compile(
            r'^\s*'
            r'(?:'
            r'\d+\.?\s+[A-Z]'
            r'|[A-Z]\.\s+[A-Z]'
            r'|(?:Abstract|Introduction|Related\s*Work|Background|Method|'
            r'Methodology|Approach|Experiment|Result|Discussion|Conclusion|'
            r'Acknowledgment|Acknowledgement|Reference|Appendix|'
            r'Implementation|Evaluation|Analysis|Limitation|Future\s*Work|'
            r'Training|Dataset|Model|Setup|Ablation)'
            r')',
            re.IGNORECASE
        )

        # Second pass: identify headers and build sections
        sections = {}
        current_section = "Preamble"
        current_text = []
        full_text_parts = []

        for page in pages_data:
            for block in page["text_blocks"]:
                text = block["text"].strip()
                font_size = block["font_size"]
                is_bold = block["is_bold"]

                if len(text) < 3:
                    continue

                is_header = False

                # Criterion 1: Larger font than body text
                if font_size > body_font_size + 0.5 and len(text) < 100:
                    is_header = True

                # Criterion 2: Bold text matching section patterns
                if is_bold and len(text) < 100 and section_pattern.match(text):
                    is_header = True

                # Criterion 3: Text matching patterns with same/larger font
                if font_size >= body_font_size and section_pattern.match(text) and len(text) < 80:
                    is_header = True

                if is_header:
                    if current_text:
                        content = "\n".join(current_text).strip()
                        if content:
                            sections[current_section] = content
                    current_section = self._clean_section_name(text)
                    current_text = []
                else:
                    current_text.append(text)

                full_text_parts.append(text)

        # Save last section
        if current_text:
            content = "\n".join(current_text).strip()
            if content:
                sections[current_section] = content

        full_text = "\n".join(full_text_parts)
        return sections, full_text

    def _clean_section_name(self, text: str) -> str:
        """Clean up a section header name."""
        cleaned = re.sub(r'^[\d\.]+\s*', '', text.strip())
        cleaned = cleaned.strip('. ')
        return cleaned if cleaned else text.strip()

    # ─── Step 3: Extract Metadata ───

    def _extract_metadata(self, pages_data: List[Dict], doc) -> tuple:
        """Extract title and authors from the first page."""
        title = doc.metadata.get("title", "").strip() if doc.metadata else ""
        author = doc.metadata.get("author", "").strip() if doc.metadata else ""

        if not title and pages_data:
            blocks = pages_data[0]["text_blocks"]
            if blocks:
                largest = max(blocks, key=lambda b: b["font_size"])
                title = re.sub(r'\s+', ' ', largest["text"].strip())
        return title, author

    def _extract_abstract(self, sections: Dict, full_text: str) -> str:
        """Extract the abstract from the paper."""
        for name, content in sections.items():
            if "abstract" in name.lower():
                return content.strip()

        match = re.search(
            r'(?:Abstract|ABSTRACT)[\s.:—-]*(.+?)(?=\n\s*\n|\n\d|Introduction|INTRODUCTION|1\s)',
            full_text,
            re.DOTALL | re.IGNORECASE
        )
        return match.group(1).strip() if match else ""

    # ─── Step 4: Chunking ───

    def _split_long_text(self, text: str) -> List[str]:
        """Split very long text into overlapping segments (character based).

        Many PDFs extract as one giant paragraph with no blank lines. Without this,
        the chunker would produce 1 chunk per section, which breaks retrieval.
        """
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size_chars:
            return [text]

        segments: List[str] = []
        max_len = int(self.chunk_size_chars)
        overlap = int(min(self.chunk_overlap_chars, max_len // 2))
        step = max(1, max_len - overlap)

        start = 0
        while start < len(text):
            end = min(start + max_len, len(text))

            # Try to end on a clean boundary near the end.
            if end < len(text):
                boundary_search_start = start + int(max_len * 0.7)
                boundary_search_start = min(boundary_search_start, end - 1)
                cut = text.rfind("\n", boundary_search_start, end)
                if cut == -1:
                    cut = text.rfind(" ", boundary_search_start, end)
                if cut != -1 and cut > start + 50:
                    end = cut

            segment = text[start:end].strip()
            if segment:
                segments.append(segment)

            # Move forward with overlap.
            if end >= len(text):
                break
            start += step

        return segments

    def _chunk_text(self, sections: Dict, pages_data: List[Dict]) -> List[TextChunk]:
        """Split sections into overlapping chunks respecting paragraph boundaries."""
        chunks = []
        chunk_id = 0
        char_offset = 0
        page_map = self._build_page_map(pages_data)

        for section_name, section_text in sections.items():
            # Skip references
            if any(kw in section_name.lower() for kw in ["reference", "bibliography"]):
                char_offset += len(section_text)
                continue

            chunk_type = "abstract" if "abstract" in section_name.lower() else "text"

            # Split into paragraphs
            paragraphs = [
                p.strip() for p in re.split(r'\n\s*\n', section_text)
                if len(p.strip()) > 20
            ]
            if not paragraphs:
                paragraphs = [section_text.strip()]

            # If paragraphs are huge (common in PDF extraction), split them into segments
            # so retrieval has enough granularity.
            expanded: List[str] = []
            for p in paragraphs:
                if len(p) > self.chunk_size_chars:
                    expanded.extend(self._split_long_text(p))
                else:
                    expanded.append(p)
            paragraphs = expanded

            # Combine paragraphs into chunks
            parts = []
            length = 0
            prev_last = ""

            for para in paragraphs:
                if length + len(para) > self.chunk_size_chars and parts:
                    text = "\n\n".join(parts)
                    pages = self._get_page_numbers(char_offset, char_offset + len(text), page_map)
                    chunks.append(TextChunk(
                        chunk_id=chunk_id, text=text, section=section_name,
                        page_numbers=pages, chunk_type=chunk_type,
                        start_char=char_offset,
                        token_count_approx=len(text) // self.chars_per_token,
                    ))
                    chunk_id += 1
                    prev_last = parts[-1] if parts else ""
                    char_offset += len(text)
                    parts = []
                    length = 0
                    if prev_last and len(prev_last) <= self.chunk_overlap_chars:
                        parts.append(prev_last)
                        length += len(prev_last)

                parts.append(para)
                length += len(para)

            # Save remaining
            if parts:
                text = "\n\n".join(parts)
                pages = self._get_page_numbers(char_offset, char_offset + len(text), page_map)
                chunks.append(TextChunk(
                    chunk_id=chunk_id, text=text, section=section_name,
                    page_numbers=pages, chunk_type=chunk_type,
                    start_char=char_offset,
                    token_count_approx=len(text) // self.chars_per_token,
                ))
                chunk_id += 1
                char_offset += len(text)

        return chunks

    def _build_page_map(self, pages_data: List[Dict]) -> List[tuple]:
        """Build character offset to page number mapping."""
        page_map = []
        offset = 0
        for page in pages_data:
            text = page["raw_text"]
            page_map.append((offset, offset + len(text), page["page_num"]))
            offset += len(text)
        return page_map

    def _get_page_numbers(self, start: int, end: int, page_map: List[tuple]) -> List[int]:
        """Given character range, return which pages it spans."""
        pages = [p for s, e, p in page_map if start < e and end > s]
        return pages if pages else [1]
