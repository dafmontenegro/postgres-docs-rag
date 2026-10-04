"""
scraper.py

Scraper for 4 chapters of the official PostgreSQL documentation.
Outputs data/raw_docs.json with a nested structure mirroring the
site's own section hierarchy.

Chapters covered:
  - Chapter 11: Indexes
  - Chapter 12: Full Text Search
  - Chapter 24: Routine Database Maintenance Tasks
  - Chapter 25: Backup and Restore

Run:
    python3 scraper.py

Each physical HTML page is fetched once and cached under cache/.
Re-runs skip already-cached pages. A 0.7s delay is added between
live requests to be polite to postgresql.org.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urldefrag

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHAPTER_ROOT_URLS = [
    "https://www.postgresql.org/docs/current/indexes.html",
    "https://www.postgresql.org/docs/current/textsearch.html",
    "https://www.postgresql.org/docs/current/maintenance.html",
    "https://www.postgresql.org/docs/current/backup.html",
]

CACHE_DIR = Path("cache")
OUTPUT_PATH = Path("data/raw_docs.json")
REQUEST_DELAY = 0.7
USER_AGENT = "pg-rag-scraper/1.0 (educational; polite)"

# Matches "11.2.1. B-Tree" -> groups ("11.2.1", "B-Tree")
SECTION_RE = re.compile(r"^(\d+(?:\.\d+)*)\.\s+(.+)$")

NOTE_CLASSES = {"note": "Note", "tip": "Tip", "caution": "Caution",
                "warning": "Warning", "important": "Important"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TocEntry:
    section_number: str
    title: str
    full_url: str       # with #fragment if present
    base_url: str       # fragment stripped
    anchor_id: Optional[str]

    @property
    def depth(self) -> int:
        return self.section_number.count(".") + 1


@dataclass
class SectionNode:
    title: str
    source_url: str
    anchor_id: Optional[str]
    content_markdown: str = ""
    subsections: dict[str, "SectionNode"] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "source_url": self.source_url,
            "anchor_id": self.anchor_id,
            "content_markdown": self.content_markdown,
            "subsections": {k: v.to_dict() for k, v in self.subsections.items()},
        }


# ---------------------------------------------------------------------------
# Fetching with disk cache
# ---------------------------------------------------------------------------

def _cache_key(url: str) -> str:
    base, _ = urldefrag(url)
    return base.split("/")[-1].replace(".html", "")


def fetch_page(url: str, session: requests.Session) -> str:
    """Fetch url, using a local HTML cache to avoid redundant requests."""
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{_cache_key(url)}.html"
    if path.exists():
        return path.read_text("utf-8")
    resp = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    path.write_text(resp.text, "utf-8")
    time.sleep(REQUEST_DELAY)
    return resp.text


# ---------------------------------------------------------------------------
# TOC parsing
# ---------------------------------------------------------------------------

def parse_toc(chapter_url: str, session: requests.Session) -> tuple[str, str, list[TocEntry]]:
    """
    Parse a chapter root page (pure TOC, no real content).
    Returns (chapter_number, chapter_title, ordered TocEntry list).
    """
    html = fetch_page(chapter_url, session)
    soup = BeautifulSoup(html, "lxml")

    h1 = soup.find("h1") or soup.find("h2")
    heading_text = h1.get_text(" ", strip=True) if h1 else ""
    m = re.match(r"Chapter\s+(\d+)\.\s*(.*)", heading_text)
    chapter_number = m.group(1) if m else ""
    chapter_title = m.group(2) if m else heading_text

    toc = soup.find(class_="toc") or soup
    entries: list[TocEntry] = []
    for a in toc.find_all("a", href=True):
        text = a.get_text(" ", strip=True)
        m2 = SECTION_RE.match(text)
        if not m2:
            continue
        sec_num, title = m2.groups()
        full_url = urljoin(chapter_url, a["href"])
        base_url, fragment = urldefrag(full_url)
        entries.append(TocEntry(
            section_number=sec_num,
            title=title.strip(),
            full_url=full_url,
            base_url=base_url,
            anchor_id=fragment or None,
        ))

    return chapter_number, chapter_title, entries


def group_pages_by_url(entries: list[TocEntry]) -> dict[str, list[TocEntry]]:
    """Group TOC entries by the physical page they live on."""
    grouped: dict[str, list[TocEntry]] = {}
    for e in entries:
        grouped.setdefault(e.base_url, []).append(e)
    return grouped


# ---------------------------------------------------------------------------
# HTML → Markdown conversion
# ---------------------------------------------------------------------------

def _table_to_markdown(table: Tag) -> str:
    rows = table.find_all("tr")
    if not rows:
        return ""
    def cells(row: Tag) -> list[str]:
        return [c.get_text(" ", strip=True).replace("|", "\\|")
                for c in row.find_all(["th", "td"])]
    header = cells(rows[0])
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows[1:]:
        lines.append("| " + " | ".join(cells(row)) + " |")
    return "\n".join(lines)


def _pre_to_markdown(pre: Tag) -> str:
    return f"```\n{pre.get_text()}\n```"


def _note_to_markdown(div: Tag, label: str) -> str:
    parts = []
    for child in div.children:
        if not isinstance(child, Tag):
            continue
        if child.name in ("h3", "strong") and child.get_text(strip=True).lower() == label.lower():
            continue   # skip the label heading itself
        if child.name == "pre":
            parts.append(_pre_to_markdown(child))
        else:
            t = child.get_text(" ", strip=True)
            if t:
                parts.append(t)
    body = "\n".join(parts)
    quoted = "\n".join(f"> {line}" for line in body.splitlines())
    return f"> **{label}**\n{quoted}"


def nodes_to_markdown(nodes: list) -> str:
    """Convert a list of DOM nodes to a Markdown string."""
    parts: list[str] = []
    for node in nodes:
        if isinstance(node, NavigableString):
            t = str(node).strip()
            if t:
                parts.append(t)
            continue
        if not isinstance(node, Tag):
            continue

        # Skip nested section containers (they are their own chunks)
        if node.name == "div" and any(
            c.startswith("sect") for c in (node.get("class") or [])
        ):
            continue

        cls_list = node.get("class") or []
        box_label = next((NOTE_CLASSES[c] for c in cls_list if c in NOTE_CLASSES), None)
        if box_label:
            parts.append(_note_to_markdown(node, box_label))
            continue

        if node.name == "pre":
            parts.append(_pre_to_markdown(node))
            continue

        if node.name in ("table",):
            parts.append(_table_to_markdown(node))
            continue

        # Inline links: keep as plain text (don't follow cross-references)
        for a in node.find_all("a", href=True):
            a.replace_with(a.get_text(" ", strip=True))

        text = node.get_text("\n", strip=True)
        if text:
            parts.append(text)

    return "\n\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Slicing a physical page by section headings
# ---------------------------------------------------------------------------

def slice_page(html: str, entries: list[TocEntry]) -> dict[str, str]:
    """
    For each entry that lives on this page, locate its section container
    and return its content as Markdown.

    PostgreSQL docs structure each section as:
        <div class="sectN" id="ANCHOR-ID">
          <div class="titlepage"><div><div>
            <hN class="title">N.M. Title <a ...>#</a></hN>
          </div></div></div>
          <p>content...</p>
          <div class="sectN+1" id="CHILD-ANCHOR">...</div>  ← sibling child sections
        </div>

    For TOC entries with an anchor_id we find the div by id directly.
    For page-level entries (no fragment in the TOC href), the main section
    is the first sect1 (or sect2/sect3) div on the page.

    In both cases we skip:
      - the titlepage div (just the heading wrapper)
      - nested sectN divs (they are their own chunks)
    """
    soup = BeautifulSoup(html, "lxml")
    result: dict[str, str] = {}

    for entry in entries:
        container = None

        if entry.anchor_id:
            container = soup.find(id=entry.anchor_id)
        else:
            # Page-level section: grab the first sectN div in the document
            for level in ("sect1", "sect2", "sect3"):
                container = soup.find("div", class_=level)
                if container:
                    break

        if container is None:
            result[entry.section_number] = ""
            continue

        children = []
        for child in container.children:
            if not isinstance(child, Tag):
                txt = str(child).strip()
                if txt:
                    children.append(child)
                continue
            cls = child.get("class") or []
            # Skip the titlepage wrapper (heading lives here, not content)
            if "titlepage" in cls:
                continue
            # Skip nested section containers — they are separate chunks
            if any(c.startswith("sect") for c in cls):
                continue
            children.append(child)

        result[entry.section_number] = nodes_to_markdown(children)

    return result


# ---------------------------------------------------------------------------
# Building the chapter tree
# ---------------------------------------------------------------------------

def build_chapter_tree(chapter_url: str, session: requests.Session) -> tuple[str, dict]:
    """Full pipeline for one chapter. Returns (chapter_title, chapter_dict)."""
    chapter_number, chapter_title, entries = parse_toc(chapter_url, session)
    pages = group_pages_by_url(entries)

    nodes: dict[str, SectionNode] = {
        e.section_number: SectionNode(
            title=e.title,
            source_url=e.full_url,
            anchor_id=e.anchor_id,
        )
        for e in entries
    }

    for base_url, page_entries in pages.items():
        html = fetch_page(base_url, session)
        slices = slice_page(html, page_entries)
        for entry in page_entries:
            nodes[entry.section_number].content_markdown = slices.get(entry.section_number, "")

    # Nest by section number depth
    top_level: dict[str, SectionNode] = {}
    for number, node in nodes.items():
        parts = number.split(".")
        if len(parts) == 1:
            top_level[number] = node
        else:
            parent_key = ".".join(parts[:-1])
            parent = nodes.get(parent_key)
            if parent:
                parent.subsections[number] = node
            else:
                top_level[number] = node   # orphan, keep visible

    return chapter_title, {
        "chapter_number": chapter_number,
        "chapter_url": chapter_url,
        "sections": {k: v.to_dict() for k, v in top_level.items()},
    }


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _count_and_words(sections: dict) -> tuple[int, int, int]:
    top = len(sections)
    subs = 0
    words = 0
    for node in sections.values():
        words += len(node.get("content_markdown", "").split())
        sub_sections = node.get("subsections", {})
        subs += len(sub_sections)
        _, deeper_subs, deeper_words = _count_and_words(sub_sections)
        subs += deeper_subs
        words += deeper_words
    return top, subs, words


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    session = requests.Session()
    result: dict[str, dict] = {}

    for url in CHAPTER_ROOT_URLS:
        print(f"  scraping {url} …")
        title, chapter_dict = build_chapter_tree(url, session)
        result[title] = chapter_dict
        top, subs, words = _count_and_words(chapter_dict["sections"])
        print(f"    chapter {chapter_dict['chapter_number']}: {top} sections, {subs} subsections, ~{words} words")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), "utf-8")

    total_top = total_subs = total_words = 0
    for c in result.values():
        t, s, w = _count_and_words(c["sections"])
        total_top += t
        total_subs += s
        total_words += w

    print("\nSummary")
    print("-------")
    print(f"Chapters:          {len(result)}")
    print(f"Top-level sections:{total_top}")
    print(f"Subsections:       {total_subs}")
    print(f"Total words:       ~{total_words}")
    print(f"Output:            {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
