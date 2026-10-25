"""The AI-generated label every picture and video difflab delivers carries.

IPTC's DigitalSourceType is what Meta, Google and Adobe read for their "AI info" labels when a file has no
C2PA manifest. A C2PA manifest needs a signing certificate, so none is written.
"""

import struct
import zlib
from pathlib import Path

SOURCE = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
SAID = f"AI-generated with difflab ({SOURCE})"

XMP = (
    '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
    '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    '<rdf:Description rdf:about="" xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/"'
    ' xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:dc="http://purl.org/dc/elements/1.1/"'
    f' Iptc4xmpExt:DigitalSourceType="{SOURCE}" xmp:CreatorTool="difflab">'
    '<dc:description><rdf:Alt><rdf:li xml:lang="x-default">AI-generated with difflab</rdf:li></rdf:Alt>'
    "</dc:description></rdf:Description></rdf:RDF></x:xmpmeta>"
    '<?xpacket end="r"?>'
)

# MP4 global tags. ffmpeg's mp4 muxer drops any other key unless `use_metadata_tags`, which moves every tag
# into QuickTime `mdta` keys that players reading the iTunes `ilst` box no longer see.
MP4_TAGS = {"comment": SAID, "description": SAID}

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
XMP_KEYWORD = b"XML:com.adobe.xmp"


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def mark_png(path: Path) -> None:
    """Spliced in after IHDR, so the pixels are not encoded again."""
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE) or XMP_KEYWORD in data[:4096]:
        return
    ihdr_end = 8 + 8 + struct.unpack(">I", data[8:12])[0] + 4
    labels = _chunk(b"iTXt", XMP_KEYWORD + b"\x00\x00\x00\x00\x00" + XMP.encode()) + _chunk(
        b"tEXt", b"Software\x00difflab"
    )
    partial = path.with_name(path.name + ".marking")
    partial.write_bytes(data[:ihdr_end] + labels + data[ihdr_end:])
    partial.replace(path)
