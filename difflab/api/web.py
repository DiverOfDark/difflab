"""The web for the chat's references: searches through `ddgs`, which reads the engines' own result pages."""

from __future__ import annotations

import io
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ddgs import DDGS
from ddgs.exceptions import DDGSException
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from difflab.api import assets

SHEET_COLUMNS = 3
CELL = 320  # pixels a result gets on the sheet: nine are 960x960, ~900 tokens to the VLM
FETCH_MOST = 40 * 1024**2
FETCH_SECONDS = 20
# Some hosts answer a library's user agent with an error page.
AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36"
FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp"}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_images",
        "description": "Pictures on the web for a query, as one numbered contact sheet with each one's address; "
        "fetch_image keeps the ones worth using.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "count": {"type": "integer", "minimum": 1, "maximum": 12, "description": "9 unless given"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_image",
        "description": "A picture from the web stored on the server; returns the path the other tools take.",
        "inputSchema": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "the picture's own address, not its page's"}},
            "required": ["url"],
        },
    },
    {
        "name": "search_web",
        "description": "Pages on the web for a query: title, address and a snippet each.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "count": {"type": "integer", "minimum": 1, "maximum": 10}},
            "required": ["query"],
        },
    },
]
NAMES = {tool["name"] for tool in TOOLS}


def call(name: str, given: dict[str, Any]) -> tuple[str, list[str]]:
    """What the tool says, and the files it hands over."""
    if name == "fetch_image":
        return _fetch(str(given["url"]))
    query = str(given["query"]).strip()
    if not query:
        raise ValueError("an empty query")
    try:
        if name == "search_images":
            return _images(query, int(given.get("count") or 9))
        return _pages(query, int(given.get("count") or 5))
    except DDGSException as problem:
        kind = "&tbm=isch" if name == "search_images" else ""
        link = f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}{kind}"
        return (
            f"The search engines refused ({problem}). Give the person this link to search in their own browser, "
            f"where they can drop what they pick into the chat: {link}",
            [],
        )


def _images(query: str, count: int) -> tuple[str, list[str]]:
    found = DDGS().images(query, safesearch="off", max_results=min(count, 12))
    if not found:
        return f"No pictures found for {query!r}.", []
    with ThreadPoolExecutor(8) as pool:
        thumbs = list(pool.map(lambda hit: _small(hit.get("thumbnail") or hit["image"]), found))
    lines = [
        f"{n}. {hit.get('title', '')[:80]} · {hit.get('width')}x{hit.get('height')} · {hit['image']}"
        + ("" if thumb else " (no preview)")
        for n, (hit, thumb) in enumerate(zip(found, thumbs, strict=True), 1)
    ]
    sheet = _sheet(thumbs)
    out = io.BytesIO()
    sheet.save(out, "JPEG", quality=88)
    stored = assets.store(out.getvalue(), "search.jpg")["path"]
    return "\n".join([f"Pictures for {query!r}, numbered as on the sheet:", *lines]), [stored]


def _pages(query: str, count: int) -> tuple[str, list[str]]:
    found = DDGS().text(query, safesearch="off", max_results=min(count, 10))
    if not found:
        return f"Nothing found for {query!r}.", []
    return "\n".join(f"{n}. {hit['title']} · {hit['href']}\n   {hit['body']}" for n, hit in enumerate(found, 1)), []


def _fetch(url: str) -> tuple[str, list[str]]:
    data = _get(url, FETCH_MOST)
    try:
        with Image.open(io.BytesIO(data)) as picture:
            picture.verify()
        with Image.open(io.BytesIO(data)) as picture:
            kind, size = picture.format, picture.size
            if kind not in FORMATS:
                converted = io.BytesIO()
                picture.convert("RGB").save(converted, "PNG")
                data, kind = converted.getvalue(), "PNG"
    except (UnidentifiedImageError, OSError) as problem:
        raise ValueError(f"{url} is not a picture ({problem}); take the picture's own address") from None
    stem = urllib.parse.urlparse(url).path.rsplit("/", 1)[-1].rsplit(".", 1)[0] or "web"
    stored = assets.store(data, f"{stem}{FORMATS[kind]}")["path"]
    return f"Stored as {stored} ({size[0]}x{size[1]}).", [stored]


def _get(url: str, most: int) -> bytes:
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        raise ValueError(f"{url}: only http and https addresses")
    request = urllib.request.Request(url, headers={"User-Agent": AGENT, "Accept": "image/*,*/*;q=0.5"})
    try:
        with urllib.request.urlopen(request, timeout=FETCH_SECONDS) as answer:
            data: bytes = answer.read(most + 1)
    except (OSError, ValueError) as problem:
        raise ValueError(f"{url} could not be fetched ({problem})") from None
    if len(data) > most:
        raise ValueError(f"{url} is over {most // 1024**2} MB")
    return data


def _small(url: str) -> Image.Image | None:
    try:
        with Image.open(io.BytesIO(_get(url, 4 * 1024**2))) as picture:
            small = picture.convert("RGB")
    except (ValueError, UnidentifiedImageError, OSError):
        return None
    small.thumbnail((CELL, CELL))
    return small


def _sheet(thumbs: list[Image.Image | None]) -> Image.Image:
    rows = (len(thumbs) + SHEET_COLUMNS - 1) // SHEET_COLUMNS
    sheet = Image.new("RGB", (SHEET_COLUMNS * CELL, rows * CELL), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=36)
    for index, thumb in enumerate(thumbs):
        x, y = index % SHEET_COLUMNS * CELL, index // SHEET_COLUMNS * CELL
        if thumb is not None:
            sheet.paste(thumb, (x + (CELL - thumb.width) // 2, y + (CELL - thumb.height) // 2))
        draw.rectangle((x, y, x + 52, y + 46), fill=(0, 0, 0))
        draw.text((x + 10, y + 4), str(index + 1), fill=(255, 220, 0), font=font)
    return sheet
