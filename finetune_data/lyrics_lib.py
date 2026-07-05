"""Shared helpers for lyrics: filename parsing, lrclib fetching with best-match
ranking, and LRC parsing. Used by 00_inventory.py, fetch_lyrics.py, 03_lyrics.py.

lrclib.net is a free, no-auth synced-lyrics database. Its /get endpoint needs a
near-exact (artist,title[,duration]) match; /search is fuzzy but returns many
hits, so we rank hits by title/artist/duration similarity instead of blindly
taking the first (which is often the wrong song).
"""
import difflib
import json
import re
import time
import urllib.parse
import urllib.request

USER_AGENT = "yue-ft-dataprep/1.0 (https://github.com/DoinJ/Song-Generator)"
CJK = "一-鿿"
LRC_TIME_RE = re.compile(r"\[(\d+):(\d+)(?:[.:](\d+))?\]")

# Cantonese lyrics on lrclib are stored in TRADITIONAL Chinese, but our filenames
# are SIMPLIFIED. Convert s->t for search + matching (huge recall gain). Falls
# back to identity if OpenCC isn't installed (then install it: see README).
try:
    from opencc import OpenCC
    _S2T = OpenCC("s2t")
    _T2S = OpenCC("t2s")
    def to_trad(s):
        return _S2T.convert(s) if s else s
    def to_simp(s):
        """Collapse all traditional variants to one simplified char for matching."""
        return _T2S.convert(s) if s else s
    HAS_OPENCC = True
except Exception:  # pragma: no cover
    def to_trad(s):
        return s
    def to_simp(s):
        return s
    HAS_OPENCC = False


# ─── filename parsing ─────────────────────────────────────────────────────────
def parse_name(stem: str):
    """Return (artist, title) from a filename stem (no extension).

    Handles: '艺人-歌名', duet '张学友_汤宝如-相思风雨中', romanized 'Beyond-海阔天空',
    and numbered outliers like '198 呼吸有害 (...) 莫文蔚'.
    """
    s = stem.strip()
    # numbered outlier: '<num> <title> <cjk-artist>' with no '-'
    m = re.match(rf"^\s*\d{{1,3}}\s+(.*?)([{CJK}]{{2,4}})\s*$", s)
    if m and "-" not in s and re.match(r"^\s*\d", s):
        return m.group(2).strip(), m.group(1).strip()
    if "-" in s:
        artist, title = s.split("-", 1)
        return artist.strip(), title.strip()
    return "", s


# ─── normalization / cleaning ─────────────────────────────────────────────────
def _norm(s: str) -> str:
    """Collapse to simplified (t2s), lowercase, strip all but letters/digits/CJK.

    t2s collapses ALL traditional variants (淒,悽) to one simplified char (凄),
    so scoring is variant-invariant. lrclib search queries still use to_trad()
    for recall — this is only for local comparison."""
    if not s:
        return ""
    return re.sub(rf"[^0-9a-z{CJK}]", "", to_simp(s).lower())


def clean_title(title: str) -> str:
    """A search-friendlier title: drop leading track no., bracketed notes, and
    live/language markers that hurt matching."""
    t = re.sub(r"^\s*\d{1,3}\s+", "", title)
    t = re.sub(r"[\(\（\[\【][^\)\）\]\】]*[\)\）\]\】]", " ", t)  # remove (…) （…） […】
    t = re.sub(r"(?i)\b(live|version|remix|feat\.?|remaster(ed)?)\b", " ", t)
    t = re.sub(r"(粤语|國語|国语|現場版|现场版|純音樂|纯音乐)", " ", t)
    return re.sub(r"\s+", " ", t).strip() or title.strip()


def clean_artist(artist: str) -> str:
    """Normalize duet separators for search."""
    return re.sub(r"[_/、&]+", " ", artist).strip()


# Interchangeable traditional variants of one simplified char. OpenCC s2t picks a
# single form (e.g. 凄->悽) but lrclib may store another (淒), so we expand search
# queries over these groups. t2s collapses them all for scoring, so once a hit is
# retrieved it still matches.
VARIANT_GROUPS = [
    set("凄淒悽"), set("裡裏"), set("台臺檯"), set("峰峯"), set("艷豔艶"),
    set("恒恆"), set("秘祕"), set("線綫"), set("污汙"), set("掛挂"), set("羣群"),
    set("詠咏"), set("裝装"), set("闖闯"), set("衆眾"), set("裊嫋"),
]
_CHAR2GROUP = {c: g for g in VARIANT_GROUPS for c in g}


def title_variants(s: str, cap: int = 5):
    """Alternate spellings of a title via single ambiguous-char substitutions."""
    out, seen = [], set()
    for i, ch in enumerate(s):
        for alt in _CHAR2GROUP.get(ch, ()):
            if alt == ch:
                continue
            v = s[:i] + alt + s[i + 1:]
            if v != s and v not in seen:
                seen.add(v)
                out.append(v)
    return out[:cap]


# ─── lrclib API ───────────────────────────────────────────────────────────────
def _http_json(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def lrclib_get(artist, title, duration=None):
    params = {"artist_name": artist, "track_name": title}
    if duration:
        params["duration"] = str(int(round(float(duration))))
    try:
        return _http_json("https://lrclib.net/api/get?" + urllib.parse.urlencode(params))
    except Exception:
        return None


def lrclib_search(q):
    try:
        return _http_json("https://lrclib.net/api/search?" + urllib.parse.urlencode({"q": q}))
    except Exception:
        return []


def score_hit(hit, title, artist, duration=None):
    """0..1 relevance score for a search hit against the wanted (title, artist, duration)."""
    ht = _norm(hit.get("trackName", ""))
    ha = _norm(hit.get("artistName", ""))
    title_r = difflib.SequenceMatcher(None, ht, _norm(clean_title(title))).ratio()
    # artist match: containment counts (romanized vs CJK often only partially overlap)
    wa = _norm(clean_artist(artist))
    if wa and ha and (wa in ha or ha in wa):
        artist_r = 1.0
    else:
        artist_r = difflib.SequenceMatcher(None, ha, wa).ratio() if wa else 0.5
    dur_r = 0.5
    if duration and hit.get("duration"):
        diff = abs(float(hit["duration"]) - float(duration))
        dur_r = max(0.0, 1.0 - diff / 15.0)  # within ~15s -> good
    score = 0.65 * title_r + 0.2 * artist_r + 0.15 * dur_r
    if hit.get("syncedLyrics"):
        score += 0.05
    return score


def fetch_best(artist, title, duration=None, min_score=0.55, sleep=0.0):
    """Return the best lrclib record for (artist, title), or None.

    Strategy:
    1. Try /get (exact match). If it returns SYNCED lyrics, return immediately.
       Otherwise keep it as a fallback and continue searching.
    2. /search with several query forms (traditional first, then simplified,
       plus title CHARACTER-VARIANT expansions for ambiguous s→t mappings like
       凄→淒/悽). Dedupe hits by id. Rank every hit with score_hit.
    3. Prefer the highest-scoring SYNCED hit. If none, fall back to the best
       plain hit (provided score >= min_score).
    4. An extra permissive pass with just the title (no artist) catches
       mislabeled artists.

    Return dict or None.
    """
    ct = clean_title(title)
    ca = clean_artist(artist)
    # traditional forms for lrclib querying (lrclib stores Cantonese in traditional)
    tt, ta = to_trad(ct), to_trad(ca)
    best_plain = None  # fallback if no synced found

    # 1) Exact get — try traditional first, keep as candidate
    rec = lrclib_get(ta, tt, duration) or lrclib_get(ca or artist, title, duration)
    if rec:
        syn = rec.get("syncedLyrics")
        pln = rec.get("plainLyrics")
        if syn:
            return _wrap_hit(rec, 1.0, "get")
        elif pln:
            best_plain = _wrap_hit(rec, 1.0, "get")

    # 2) Fuzzy search with query forms, including character-variant expansions.
    # Base queries (traditional first for recall)
    base_queries = [f"{ta} {tt}".strip(), tt, f"{ca} {ct}".strip(), ct]
    # Add title-variant queries (artist + variant-title)
    for tv in title_variants(tt) + title_variants(ct):
        base_queries.append(f"{ta} {tv}".strip())
        base_queries.append(tv)
    # Permissive title-only queries catch wrong/missing artist labels
    if ta and tt:
        base_queries.append(f"{ta}")
    seen, all_hits = set(), []
    for q in dict.fromkeys([q for q in base_queries if q]):
        for h in lrclib_search(q):
            hid = h.get("id")
            if hid in seen:
                continue
            seen.add(hid)
            all_hits.append(h)
        if sleep:
            time.sleep(sleep)

    if not all_hits:
        return best_plain  # fall back to plain-only get result

    # Rank hits. Prefer synced: if any synced hit is within 0.1 of the overall
    # best score (or >= min_score) pick the top synced.
    synced = [h for h in all_hits if h.get("syncedLyrics")]
    if synced:
        best_sync = max(synced, key=lambda h: score_hit(h, title, artist, duration))
        sync_score = score_hit(best_sync, title, artist, duration)
        if sync_score >= min_score:
            return _wrap_hit(best_sync, sync_score, "search")
    # No synced found — pick the best plain hit (which could also be the get fallback)
    if all_hits:
        best = max(all_hits, key=lambda h: score_hit(h, title, artist, duration))
        bscore = score_hit(best, title, artist, duration)
        if bscore >= min_score and best.get("plainLyrics"):
            return _wrap_hit(best, bscore, "search")
    return best_plain  # last resort: plain get


def _wrap_hit(hit, score, source):
    return {
        "syncedLyrics": hit.get("syncedLyrics"),
        "plainLyrics": hit.get("plainLyrics"),
        "matched_artist": hit.get("artistName"),
        "matched_track": hit.get("trackName"),
        "matched_duration": hit.get("duration"),
        "score": round(score, 3),
        "source": source,
    }


# ─── LRC parsing ──────────────────────────────────────────────────────────────
def parse_lrc(text):
    """Parse LRC text -> sorted [(time_sec, line_text)] with non-empty text."""
    lines = []
    for raw in text.splitlines():
        stamps = LRC_TIME_RE.findall(raw)
        if not stamps:
            continue
        content = LRC_TIME_RE.sub("", raw).strip()
        if not content:
            continue
        for mm, ss, frac in stamps:
            t = int(mm) * 60 + int(ss)
            if frac:
                t += int(frac) / (10 ** len(frac))
            lines.append((t, content))
    lines.sort(key=lambda x: x[0])
    return lines
