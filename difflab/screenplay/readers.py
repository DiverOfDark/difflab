"""A story file as markdown text: headings as `#` by level, paragraphs apart. EbookLib is AGPL, and an
epub and a docx are zips of XML, so those two are read here; a pdf goes through pypdf (BSD)."""

from __future__ import annotations

import io
import posixpath
import re
import zipfile
from pathlib import PurePosixPath
from xml.etree import ElementTree

import pypdf

KINDS = (".epub", ".pdf", ".txt", ".md", ".docx")
_HEADING_TAG = re.compile(r"h([1-6])")
_DOCX = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_BLOCKS = {"p", "div", "li", "blockquote", "pre", "tr"}


def read(data: bytes, filename: str) -> str:
    suffix = PurePosixPath(filename.lower()).suffix
    if suffix not in KINDS:
        raise ValueError(f"{filename}: a story is one of {', '.join(KINDS)}")
    try:
        if suffix == ".epub":
            return epub(data)
        if suffix == ".docx":
            return docx(data)
        if suffix == ".pdf":
            return pdf(data)
    except (zipfile.BadZipFile, ElementTree.ParseError, KeyError, pypdf.errors.PyPdfError) as problem:
        raise ValueError(f"{filename} could not be read: {problem}") from problem
    return data.decode("utf-8", errors="replace").replace("\r\n", "\n")


def epub(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as book:
        container = ElementTree.fromstring(book.read("META-INF/container.xml"))
        package = next(e.get("full-path", "") for e in container.iter() if _local(e.tag) == "rootfile")
        opf = ElementTree.fromstring(book.read(package))
        items = {e.get("id"): e.get("href", "") for e in opf.iter() if _local(e.tag) == "item"}
        spine = [items[e.get("idref")] for e in opf.iter() if _local(e.tag) == "itemref" and e.get("idref") in items]
        folder = posixpath.dirname(package)
        parts = [_xhtml(book.read(posixpath.normpath(posixpath.join(folder, href.split("#")[0])))) for href in spine]
    return _tidy("\n\n".join(parts))


def _xhtml(data: bytes) -> str:
    root = ElementTree.fromstring(data)
    body = next((e for e in root.iter() if _local(e.tag) == "body"), root)
    out: list[str] = []
    _walk(body, out)
    return "\n\n".join(out)


def _walk(element: ElementTree.Element, out: list[str]) -> None:
    for child in element:
        tag = _local(child.tag)
        heading = _HEADING_TAG.fullmatch(tag)
        if heading:
            out.append("#" * int(heading.group(1)) + " " + _words(child))
        elif tag in _BLOCKS and not any(
            _local(c.tag) in _BLOCKS or _HEADING_TAG.fullmatch(_local(c.tag)) for c in child
        ):
            out.append(_words(child))
        else:
            _walk(child, out)
            if child.tail and child.tail.strip():
                out.append(" ".join(child.tail.split()))


def docx(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as document:
        root = ElementTree.fromstring(document.read("word/document.xml"))
        styles = _docx_styles(document)
    out = []
    for paragraph in root.iter(f"{_DOCX}p"):
        said = "".join(t.text or "" for t in paragraph.iter(f"{_DOCX}t"))
        style = paragraph.find(f"{_DOCX}pPr/{_DOCX}pStyle")
        level = _docx_level(styles.get(style.get(f"{_DOCX}val", ""), "") if style is not None else "")
        out.append(("#" * level + " " if level and said.strip() else "") + said)
    return _tidy("\n\n".join(out))


def _docx_styles(document: zipfile.ZipFile) -> dict[str, str]:
    """Style id -> its name: a translated Word names its heading styles by id ("Überschrift1") but not by name."""
    if "word/styles.xml" not in document.namelist():
        return {}
    root = ElementTree.fromstring(document.read("word/styles.xml"))
    out = {}
    for style in root.iter(f"{_DOCX}style"):
        name = style.find(f"{_DOCX}name")
        out[style.get(f"{_DOCX}styleId", "")] = name.get(f"{_DOCX}val", "") if name is not None else ""
    return out


def _docx_level(name: str) -> int:
    lowered = name.lower().replace(" ", "")
    if lowered == "title":
        return 1
    found = re.fullmatch(r"heading([1-6])", lowered)
    return int(found.group(1)) if found else 0


def pdf(data: bytes) -> str:
    """Each page's text; with an outline, its top entries become `#` headings where their pages begin."""
    reader = pypdf.PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    starts: dict[int, list[str]] = {}
    for entry in reader.outline:
        if isinstance(entry, list):
            continue
        page = reader.get_destination_page_number(entry)
        if page is not None and 0 <= page < len(pages):
            starts.setdefault(page, []).append(" ".join(str(entry.title).split()))
    out = []
    for number, text in enumerate(pages):
        out += [f"# {title}" for title in starts.get(number, [])]
        out.append(re.sub(r"(?<!\n)\n(?!\n)", " ", text))
    return _tidy("\n\n".join(out))


def _words(element: ElementTree.Element) -> str:
    return " ".join("".join(element.itertext()).split())


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _tidy(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", "\n\n".join(p.strip() for p in text.split("\n\n") if p.strip()))
