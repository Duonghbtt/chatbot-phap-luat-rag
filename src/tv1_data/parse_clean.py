from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from html import unescape
from pathlib import Path
from typing import Any

try:
    from bs4 import BeautifulSoup, Tag
except ImportError:  # pragma: no cover - handled explicitly at runtime.
    BeautifulSoup = None  # type: ignore[assignment]
    Tag = Any  # type: ignore[misc,assignment]

LOGGER = logging.getLogger(__name__)

LAW_ID_PATTERN = re.compile(
    r"(Bộ luật số\s+[^\s,()]+|"
    r"Luật số\s+[^\s,()]+|"
    r"Pháp lệnh số\s+[^\s,()]+|"
    r"Nghị quyết liên tịch số\s+[^\s,()]+|"
    r"Nghị quyết số\s+[^\s,()]+|"
    r"Nghị định số\s+[^\s,()]+|"
    r"Thông tư liên tịch số\s+[^\s,()]+|"
    r"Thông tư số\s+[^\s,()]+|"
    r"Quyết định số\s+[^\s,()]+|"
    r"Chỉ thị số\s+[^\s,()]+|"
    r"Lệnh số\s+[^\s,()]+)",
    re.IGNORECASE,
)
ARTICLE_TITLE_PATTERN = re.compile(
    r"^(Điều\s+[0-9A-Za-zÀ-ỹ.\-]+\.)\s*(.*)$",
    re.IGNORECASE,
)
ORIGINAL_ARTICLE_PATTERN = re.compile(r"\b(Điều\s+\d+[A-Za-zÀ-ỹ\-]*)\b", re.IGNORECASE)
CHAPTER_CODE_PATTERN = re.compile(r"^Chương\s+[IVXLCDM0-9]+$", re.IGNORECASE)
DATE_PATTERNS = (
    re.compile(r"có hiệu lực thi hành kể từ ngày\s+(\d{1,2}/\d{1,2}/\d{4})", re.IGNORECASE),
    re.compile(r"có hiệu lực kể từ ngày\s+(\d{1,2}/\d{1,2}/\d{4})", re.IGNORECASE),
    re.compile(r"có hiệu lực từ ngày\s+(\d{1,2}/\d{1,2}/\d{4})", re.IGNORECASE),
)
TEXTUAL_DATE_PATTERNS = (
    re.compile(
        r"có hiệu lực thi hành kể từ ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
        re.IGNORECASE,
    ),
    re.compile(
        r"có hiệu lực kể từ ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
        re.IGNORECASE,
    ),
    re.compile(
        r"có hiệu lực từ ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
        re.IGNORECASE,
    ),
)
SOURCE_NOTE_DATE_PATTERN = re.compile(
    r"ngày\s+(?:\d{1,2}/\d{1,2}/\d{4}|\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})",
    re.IGNORECASE,
)
ISSUER_PATTERNS = (
    "Chủ tịch nước",
    "Ủy ban Thường vụ Quốc hội",
    "Quốc hội",
    "Chính phủ",
    "Thủ tướng Chính phủ",
    "Bộ Công an",
    "Bộ Quốc phòng",
    "Bộ Nội vụ",
    "Bộ Tư pháp",
    "Bộ Tài chính",
    "Bộ Kế hoạch và Đầu tư",
    "Bộ Công Thương",
    "Bộ Nông nghiệp và Phát triển nông thôn",
    "Bộ Giao thông vận tải",
    "Bộ Xây dựng",
    "Bộ Tài nguyên và Môi trường",
    "Bộ Thông tin và Truyền thông",
    "Bộ Giáo dục và Đào tạo",
    "Bộ Y tế",
    "Bộ Lao động - Thương binh và Xã hội",
    "Bộ Văn hóa, Thể thao và Du lịch",
    "Bộ Khoa học và Công nghệ",
    "Bộ Ngoại giao",
    "Ngân hàng Nhà nước Việt Nam",
    "Tòa án nhân dân tối cao",
    "Viện kiểm sát nhân dân tối cao",
    "Ủy ban nhân dân",
    "Hội đồng nhân dân",
)

ISSUER_CODE_MAP = {
    "QH": "Quốc hội",
    "UBTVQH": "Ủy ban Thường vụ Quốc hội",
    "CP": "Chính phủ",
    "TTG": "Thủ tướng Chính phủ",
    "HĐTP": "Hội đồng Thẩm phán Tòa án nhân dân tối cao",
    "CTN": "Chủ tịch nước",
    "BCA": "Bộ Công an",
    "BQP": "Bộ Quốc phòng",
    "BNV": "Bộ Nội vụ",
    "BTP": "Bộ Tư pháp",
    "TC": "Bộ Tài chính",
    "TCT": "Tổng cục Thuế",
    "BTC": "Bộ Tài chính",
    "BKHĐT": "Bộ Kế hoạch và Đầu tư",
    "BCT": "Bộ Công Thương",
    "BNN": "Bộ Nông nghiệp và Phát triển nông thôn",
    "BNNPTTN": "Bộ Nông nghiệp và Phát triển nông thôn",
    "BNNPTNT": "Bộ Nông nghiệp và Phát triển nông thôn",
    "BGTVT": "Bộ Giao thông vận tải",
    "BXD": "Bộ Xây dựng",
    "BTNMT": "Bộ Tài nguyên và Môi trường",
    "BTTTT": "Bộ Thông tin và Truyền thông",
    "BGDĐT": "Bộ Giáo dục và Đào tạo",
    "BYT": "Bộ Y tế",
    "BLĐTBXH": "Bộ Lao động - Thương binh và Xã hội",
    "BVHTTDL": "Bộ Văn hóa, Thể thao và Du lịch",
    "BKHCN": "Bộ Khoa học và Công nghệ",
    "BNG": "Bộ Ngoại giao",
    "NHNN": "Ngân hàng Nhà nước Việt Nam",
    "UBTDTT": "Ủy ban Thể dục thể thao",
    "TANDTC": "Tòa án nhân dân tối cao",
    "VKSNDTC": "Viện kiểm sát nhân dân tối cao",
    "UBND": "Ủy ban nhân dân",
    "HĐND": "Hội đồng nhân dân",
}


class MissingDependencyError(RuntimeError):
    """Raised when a required local parsing dependency is unavailable."""


@dataclass(slots=True)
class TopicInfo:
    """Topic metadata loaded from `jsonData.js`."""

    topic_id: str
    name: str
    ordinal: str = ""


@dataclass(slots=True)
class DeMucInfo:
    """Đề mục metadata loaded from `jsonData.js`."""

    file_id: str
    name: str
    topic_id: str
    ordinal: str = ""


@dataclass(slots=True)
class CorpusLookup:
    """Lookup tables derived from the official offline snapshot."""

    demuc_by_file_id: dict[str, DeMucInfo] = field(default_factory=dict)
    topic_by_id: dict[str, TopicInfo] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "CorpusLookup":
        return cls()

    def get_topic(self, topic_id: str) -> TopicInfo | None:
        return self.topic_by_id.get(topic_id)

    def get_demuc(self, file_id: str) -> DeMucInfo | None:
        return self.demuc_by_file_id.get(file_id)


@dataclass(slots=True)
class ParsedArticle:
    """Normalized representation of one Điều within a Bộ pháp điển file."""

    file_id: str
    source_path: str
    de_muc: str
    topic_id: str
    topic_name: str
    article_code: str
    article_name: str
    article: str
    mapc: str
    law_id: str
    title: str
    issuer: str
    effective_date: str
    source_note: str
    related_articles: list[str]
    raw_content: str
    chapter_code: str = ""
    chapter_title: str = ""

    @property
    def heading(self) -> str:
        return " ".join(part for part in (self.article_code, self.article_name) if part).strip()


@dataclass(slots=True)
class ParsedDocument:
    """Parsed representation of one local HTML source file."""

    source_path: Path
    file_id: str
    de_muc: str
    topic_id: str
    topic_name: str
    articles: list[ParsedArticle] = field(default_factory=list)


@dataclass(slots=True)
class SourceMetadata:
    """Metadata extracted from a `pGhiChu` source note."""

    law_id: str = ""
    title: str = ""
    article: str = ""
    issuer: str = ""
    effective_date: str = ""


def _require_bs4() -> None:
    if BeautifulSoup is None:
        raise MissingDependencyError(
            "BeautifulSoup4 is required for TV1 HTML parsing. Install `beautifulsoup4` in the local Python environment."
        )


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""

    raw = unescape(text or "")
    raw = raw.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")

    normalized_lines: list[str] = []
    previous_blank = False
    for line in raw.split("\n"):
        cleaned = re.sub(r"[ \t\f\v]+", " ", line).strip()
        if not cleaned:
            if not previous_blank:
                normalized_lines.append("")
            previous_blank = True
            continue

        normalized_lines.append(cleaned)
        previous_blank = False

    return "\n".join(normalized_lines).strip()


def _extract_js_array(js_text: str, var_name: str) -> list[dict[str, Any]]:
    pattern = re.compile(rf"var\s+{re.escape(var_name)}\s*=\s*")
    match = pattern.search(js_text)
    if not match:
        return []

    start_index = js_text.find("[", match.end())
    if start_index < 0:
        return []

    depth = 0
    in_string = False
    escape = False
    end_index = -1
    for index in range(start_index, len(js_text)):
        char = js_text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                end_index = index
                break

    if end_index < 0:
        return []

    return json.loads(js_text[start_index : end_index + 1])


def load_corpus_lookup(snapshot_root: Path | None, logger: logging.Logger | None = None) -> CorpusLookup:
    """Load `jdChuDe` and `jdDeMuc` lookups from the offline snapshot, if present."""

    log = logger or LOGGER
    if snapshot_root is None:
        return CorpusLookup.empty()

    js_path = snapshot_root / "jsonData.js"
    if not js_path.exists():
        log.warning("Không tìm thấy %s để ánh xạ topic_id/de_muc; parser sẽ tiếp tục với metadata rỗng.", js_path)
        return CorpusLookup.empty()

    js_text = js_path.read_text(encoding="utf-8")
    topic_rows = _extract_js_array(js_text, "jdChuDe")
    demuc_rows = _extract_js_array(js_text, "jdDeMuc")

    lookup = CorpusLookup(
        demuc_by_file_id={
            row["Value"]: DeMucInfo(
                file_id=row["Value"],
                name=clean_text(row.get("Text", "")),
                topic_id=row.get("ChuDe", ""),
                ordinal=str(row.get("STT", "")),
            )
            for row in demuc_rows
            if row.get("Value")
        },
        topic_by_id={
            row["Value"]: TopicInfo(
                topic_id=row["Value"],
                name=clean_text(row.get("Text", "")),
                ordinal=str(row.get("STT", "")),
            )
            for row in topic_rows
            if row.get("Value")
        },
    )
    return lookup


def _extract_de_muc_label(soup: "BeautifulSoup", file_id: str, lookup: CorpusLookup) -> str:
    h3 = soup.find("h3")
    if h3:
        lines = [clean_text(text) for text in h3.stripped_strings if clean_text(text)]
        if len(lines) >= 2:
            return f"{lines[0]} - {lines[1]}"
        if lines:
            return " - ".join(lines)

    demuc_info = lookup.get_demuc(file_id)
    if demuc_info:
        return demuc_info.name
    return file_id


def parse_article_title(title: str) -> tuple[str, str]:
    """Split `Điều ...` heading into code and human-readable article name."""

    normalized = clean_text(title)
    match = ARTICLE_TITLE_PATTERN.match(normalized)
    if not match:
        return normalized, ""
    return match.group(1).strip(), match.group(2).strip()


def _detect_issuer_fallback(note: str) -> str:
    for issuer in ISSUER_PATTERNS:
        if issuer in note:
            return issuer
    return ""


def _normalize_code_tokens(value: str) -> list[str]:
    normalized = clean_text(value).upper()
    normalized = re.sub(r"\s*-\s*", "-", normalized)
    normalized = normalized.replace(".", "-")
    return [token for token in re.split(r"[^0-9A-ZÀ-ỸĐ-]+", normalized) if token]


def _infer_issuer_from_compact_text(value: str) -> str:
    normalized = clean_text(value).upper()
    if not normalized:
        return ""

    inferred: list[str] = []
    for token in _normalize_code_tokens(value):
        token_parts = [part for part in token.split("-") if part]
        for token_part in token_parts:
            issuer = ""
            if token_part.startswith("UBTVQH"):
                issuer = ISSUER_CODE_MAP["UBTVQH"]
            elif re.fullmatch(r"QH\d+", token_part) or token_part == "QH":
                issuer = ISSUER_CODE_MAP["QH"]
            elif token_part in ISSUER_CODE_MAP:
                issuer = ISSUER_CODE_MAP[token_part]

            if issuer and issuer not in inferred:
                inferred.append(issuer)

    if inferred:
        return "; ".join(inferred)

    return ""


def _infer_issuer_from_note_structure(note: str) -> str:
    normalized = clean_text(note)
    lowered = normalized.lower()
    if not normalized:
        return ""

    structural_view = re.sub(r"^điều\s+[0-9a-zà-ỹ.\-]+\s+", "", lowered, flags=re.IGNORECASE).strip()
    if structural_view.startswith("pháp lệnh"):
        return "Ủy ban Thường vụ Quốc hội"
    if structural_view.startswith("lệnh"):
        return "Chủ tịch nước"
    if structural_view.startswith("nghị định"):
        return "Chính phủ"
    if structural_view.startswith("luật") or structural_view.startswith("bộ luật"):
        return "Quốc hội"
    if structural_view.startswith("quyết định") and "-ttg" in normalized.lower():
        return "Thủ tướng Chính phủ"
    if structural_view.startswith("nghị quyết") and ("hđtp" in normalized.lower() or "hội đồng thẩm phán" in normalized.lower()):
        return "Hội đồng Thẩm phán Tòa án nhân dân tối cao"

    return ""


def _infer_issuer_from_law_context(law_id: str, title: str, note: str) -> str:
    for value in (law_id, title):
        issuer = _infer_issuer_from_compact_text(value)
        if issuer:
            return issuer

    issuer = _infer_issuer_from_note_structure(note)
    if issuer:
        return issuer

    normalized = clean_text(law_id).upper()
    if normalized.startswith("BỘ LUẬT SỐ") or normalized.startswith("LUẬT SỐ"):
        return "Quốc hội"
    if normalized.startswith("PHÁP LỆNH SỐ"):
        return "Ủy ban Thường vụ Quốc hội"
    if normalized.startswith("LỆNH SỐ"):
        return "Chủ tịch nước"
    if normalized.startswith("NGHỊ ĐỊNH SỐ"):
        return "Chính phủ"
    if normalized.startswith("NGHỊ QUYẾT SỐ") and "HĐND" in normalized:
        return "Hội đồng nhân dân"
    if normalized.startswith("NGHỊ QUYẾT SỐ") and "QH" in normalized:
        return "Quốc hội"
    if normalized.startswith("QUYẾT ĐỊNH SỐ") and "TTG" in normalized:
        return "Thủ tướng Chính phủ"
    if normalized.startswith("CHỈ THỊ SỐ") and "TTG" in normalized:
        return "Thủ tướng Chính phủ"

    return ""


def _extract_effective_date(note: str) -> str:
    for pattern in DATE_PATTERNS:
        match = pattern.search(note)
        if match:
            return match.group(1)

    for pattern in TEXTUAL_DATE_PATTERNS:
        match = pattern.search(note)
        if match:
            day, month, year = match.groups()
            return f"{int(day):02d}/{int(month):02d}/{year}"

    return ""


def parse_source_note(source_note: str) -> SourceMetadata:
    """Extract source-document metadata from a Bộ pháp điển source note."""

    note = clean_text(source_note).strip("() ")
    if not note:
        return SourceMetadata()

    article_match = ORIGINAL_ARTICLE_PATTERN.search(note)
    article = article_match.group(1).strip() if article_match else ""

    law_match = LAW_ID_PATTERN.search(note)
    law_id = law_match.group(1).strip() if law_match else ""

    title = ""
    if law_match:
        tail = note[law_match.end() :].strip()
        title_end_markers = []
        for marker in (
            SOURCE_NOTE_DATE_PATTERN.search(tail),
            re.search(r",\s*có hiệu lực", tail, re.IGNORECASE),
            re.search(r",\s*hết hiệu lực", tail, re.IGNORECASE),
        ):
            if marker:
                title_end_markers.append(marker.start())
        end_index = min(title_end_markers) if title_end_markers else len(tail)
        title = tail[:end_index].strip(" ,;:-")

    issuer = ""
    before_effective = re.split(r",\s*(?:có hiệu lực|hết hiệu lực)", note, maxsplit=1, flags=re.IGNORECASE)[0]
    date_matches = list(SOURCE_NOTE_DATE_PATTERN.finditer(before_effective))
    if date_matches:
        issuer_tail = before_effective[date_matches[-1].end() :].strip()
        if issuer_tail.lower().startswith("của "):
            issuer = clean_text(issuer_tail[4:])
    if not issuer:
        issuer = _detect_issuer_fallback(note)
    if not issuer:
        issuer = _infer_issuer_from_law_context(law_id, title, note)

    effective_date = _extract_effective_date(note)

    return SourceMetadata(
        law_id=law_id,
        title=title,
        article=article,
        issuer=issuer,
        effective_date=effective_date,
    )


def _extract_related_articles(related_tags: list[Tag]) -> list[str]:
    related_articles: list[str] = []
    seen: set[str] = set()

    for tag in related_tags:
        for anchor in tag.find_all("a"):
            label = clean_text(anchor.get_text(" ", strip=True))
            if not label or label in seen:
                continue
            related_articles.append(label)
            seen.add(label)

    return related_articles


def _extract_content_text(content_tag: Tag) -> str:
    paragraphs = [
        clean_text(paragraph.get_text(" ", strip=False))
        for paragraph in content_tag.find_all("p")
    ]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    if paragraphs:
        return "\n".join(paragraphs)

    return clean_text(content_tag.get_text("\n", strip=False))


def _prefer_longer(current: str, candidate: str) -> str:
    if candidate and len(candidate) > len(current):
        return candidate
    return current


def _is_clean_issuer(value: str) -> bool:
    lowered = value.lower()
    return bool(value) and "ngày" not in lowered and len(value) <= 80


def _prefer_issuer(current: str, candidate: str) -> str:
    if not candidate:
        return current
    if not current:
        return candidate
    if _is_clean_issuer(candidate) and not _is_clean_issuer(current):
        return candidate
    if _is_clean_issuer(current) and not _is_clean_issuer(candidate):
        return current
    return current


def _backfill_document_metadata(articles: list[ParsedArticle]) -> None:
    source_cache: dict[str, SourceMetadata] = {}

    for article in articles:
        if not article.law_id:
            continue

        cached = source_cache.setdefault(article.law_id, SourceMetadata(law_id=article.law_id))
        cached.title = _prefer_longer(cached.title, article.title)
        cached.issuer = _prefer_issuer(cached.issuer, article.issuer)
        cached.effective_date = _prefer_longer(cached.effective_date, article.effective_date)
        cached.article = _prefer_longer(cached.article, article.article)

    for article in articles:
        if not article.law_id:
            continue

        cached = source_cache.get(article.law_id)
        if cached is None:
            continue

        article.title = article.title or cached.title
        article.issuer = article.issuer or cached.issuer
        article.effective_date = article.effective_date or cached.effective_date
        article.article = article.article or cached.article


def parse_html_file(
    source_path: Path,
    lookup: CorpusLookup | None = None,
    logger: logging.Logger | None = None,
) -> ParsedDocument:
    """Parse one local HTML source file from Bộ pháp điển điện tử."""

    _require_bs4()
    log = logger or LOGGER
    corpus_lookup = lookup or CorpusLookup.empty()

    html = source_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    file_id = source_path.stem
    demuc_info = corpus_lookup.get_demuc(file_id)
    topic_id = demuc_info.topic_id if demuc_info else ""
    topic_info = corpus_lookup.get_topic(topic_id) if topic_id else None
    topic_name = topic_info.name if topic_info else ""
    de_muc = _extract_de_muc_label(soup, file_id, corpus_lookup)

    content_root = soup.find("div", class_="_content") or soup
    current_chapter_code = ""
    current_chapter_title = ""
    articles: list[ParsedArticle] = []

    for node in content_root.children:
        if getattr(node, "name", None) != "p":
            continue

        classes = set(node.get("class", []))
        if "pChuong" in classes:
            chapter_text = clean_text(node.get_text(" ", strip=False))
            if not chapter_text:
                continue

            if CHAPTER_CODE_PATTERN.match(chapter_text):
                current_chapter_code = chapter_text
                current_chapter_title = ""
            else:
                current_chapter_title = chapter_text
            continue

        if "pDieu" not in classes:
            continue

        article_title = clean_text(node.get_text(" ", strip=False))
        article_code, article_name = parse_article_title(article_title)
        anchor = node.find("a")
        mapc = anchor.get("name", "") if anchor else ""

        source_note = ""
        raw_content = ""
        related_tags: list[Tag] = []
        cursor = node.next_sibling
        while cursor is not None:
            if getattr(cursor, "name", None) != "p":
                cursor = cursor.next_sibling
                continue

            sibling_classes = set(cursor.get("class", []))
            if "pDieu" in sibling_classes or "pChuong" in sibling_classes:
                break

            if "pGhiChu" in sibling_classes and not source_note:
                source_note = clean_text(cursor.get_text(" ", strip=False))
            elif "pNoiDung" in sibling_classes and not raw_content:
                raw_content = _extract_content_text(cursor)
            elif "pChiDan" in sibling_classes:
                related_tags.append(cursor)

            cursor = cursor.next_sibling

        source_metadata = parse_source_note(source_note)
        article = ParsedArticle(
            file_id=file_id,
            source_path=str(source_path.resolve()),
            de_muc=de_muc,
            topic_id=topic_id,
            topic_name=topic_name,
            article_code=article_code,
            article_name=article_name,
            article=source_metadata.article or article_code,
            mapc=mapc,
            law_id=source_metadata.law_id,
            title=source_metadata.title,
            issuer=source_metadata.issuer,
            effective_date=source_metadata.effective_date,
            source_note=source_note,
            related_articles=_extract_related_articles(related_tags),
            raw_content=raw_content,
            chapter_code=current_chapter_code,
            chapter_title=current_chapter_title,
        )
        articles.append(article)

    _backfill_document_metadata(articles)

    for article in articles:
        if not article.issuer:
            log.warning(
                "Thiếu issuer ở file %s, %s (%s)",
                source_path.name,
                article.article_code or article.mapc,
                article.law_id or "không rõ law_id",
            )
        if not article.effective_date:
            log.warning(
                "Thiếu effective_date ở file %s, %s (%s)",
                source_path.name,
                article.article_code or article.mapc,
                article.law_id or "không rõ law_id",
            )

    return ParsedDocument(
        source_path=source_path.resolve(),
        file_id=file_id,
        de_muc=de_muc,
        topic_id=topic_id,
        topic_name=topic_name,
        articles=articles,
    )


__all__ = [
    "CorpusLookup",
    "DeMucInfo",
    "MissingDependencyError",
    "ParsedArticle",
    "ParsedDocument",
    "SourceMetadata",
    "TopicInfo",
    "clean_text",
    "load_corpus_lookup",
    "parse_article_title",
    "parse_html_file",
    "parse_source_note",
]
