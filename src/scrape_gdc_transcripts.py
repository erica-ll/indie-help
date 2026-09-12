"""Pull transcripts from GDC's official YouTube channel (@Gdconf) and write
them as .txt files into RAW_DIR for text_processor.py to pick up.

Uses YouTube's own caption tracks via youtube-transcript-api -- not audio
download + re-transcription -- since GDC deliberately posts these talks for
free public viewing (distinct from GDC Vault's paywalled archive), and the
caption track is the lowest-friction way to get at that public content.
Channel video listing uses yt-dlp with extract_flat=True, which only reads
the channel's video-list metadata and never downloads any video/audio.

Usage:
    python scrape_gdc_transcripts.py --max 150
    python scrape_gdc_transcripts.py --max 300 --keywords narrative,economy,indie
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, RequestBlocked
from youtube_transcript_api.proxies import WebshareProxyConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DIR

CHANNEL_URL = "https://www.youtube.com/channel/UC0JB7TSe49lg56u6qH8y_MQ/videos"
# YouTube's anti-bot IP block kicks in after ~30-40 requests at 1s spacing --
# confirmed empirically (first run: 34 succeeded, then every remaining
# request failed with RequestBlocked). Space requests out much more, and
# treat a block as a signal to cool down hard rather than a per-video error.
REQUEST_DELAY_SECONDS = 8.0
BLOCK_COOLDOWN_SECONDS = 120
MAX_BLOCK_RETRIES = 2

# Persistent record of {video_id: title} -- skips a video whose id we've
# already fetched, even if RAW_DIR itself was reorganized (files moved into
# a subfolder, etc.) since then and the old filename-existence check alone
# wouldn't catch it.
IDS_MANIFEST_PATH = RAW_DIR / ".scraped_ids.json"

# Minimum fraction of distinctive title words that must appear somewhere in
# the fetched transcript. Best-effort guard against one quarantine_dupes
# failure mode: yt-dlp's flat channel-listing extraction occasionally
# mis-pairs a video's id with a different video's title, so the transcript
# fetched for that id has nothing to do with the title it gets saved under.
TITLE_MATCH_THRESHOLD = 0.25
_STOPWORDS = {"a", "an", "the", "of", "in", "on", "to", "for", "and", "or",
              "with", "your", "how", "is", "are", "from", "at", "by", "this"}

# Confirmed by diffing DB/raw/quarantine_dupes: GDC's channel genuinely
# hosts the same talk's captions under two entirely different video ids and
# titles (not a retitle -- retitling doesn't change a video's id, and these
# still show up as two distinct ids in the current channel listing). Neither
# the id manifest nor the title/content check catches this, since each copy
# has a legitimately different id and a title that legitimately matches its
# own content -- only comparing the actual transcript text against what's
# already on disk catches it.
QUARANTINE_DIR = RAW_DIR / "quarantine_dupes"


def _hash_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_existing_content_hashes():
    """Hashes every transcript already in the live corpus (RAW_DIR's direct
    .txt children) and everything already sitting in QUARANTINE_DIR, so a
    newly-fetched transcript that's byte-identical to either gets routed
    straight to quarantine instead of duplicating it back into the live
    corpus."""
    hashes = {}
    for path in list(RAW_DIR.glob("GDC - *.txt")) + list(QUARANTINE_DIR.glob("*.txt")):
        hashes[_hash_text(path.read_text())] = path
    return hashes


def load_scraped_ids():
    if IDS_MANIFEST_PATH.exists():
        return json.loads(IDS_MANIFEST_PATH.read_text())
    return {}


def save_scraped_ids(scraped_ids):
    IDS_MANIFEST_PATH.write_text(json.dumps(scraped_ids, indent=2, sort_keys=True))


def title_matches_content(title, text):
    """False means the transcript likely doesn't belong to this title (see
    IDS_MANIFEST_PATH comment above) -- caller should flag it instead of
    silently saving it into the live corpus."""
    words = [w.lower() for w in re.findall(r"[A-Za-z']{4,}", title) if w.lower() not in _STOPWORDS]
    if not words:
        return True
    text_lower = text.lower()
    hits = sum(1 for w in words if w in text_lower)
    return (hits / len(words)) >= TITLE_MATCH_THRESHOLD


def list_channel_videos(max_videos):
    """Returns [{"id": ..., "title": ...}, ...]. extract_flat=True means this
    only reads the channel's listing page -- no video/audio is downloaded."""
    ydl_opts = {
        "extract_flat": True,
        "quiet": True,
        "playlist_items": f"1-{max_videos}",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(CHANNEL_URL, download=False)
    return [{"id": e["id"], "title": e["title"]} for e in info["entries"] if e and e.get("id")]


def sanitize_filename(title):
    cleaned = re.sub(r'[\\/:*?"<>|]', "", title).strip()
    return cleaned[:150]


def fetch_transcript_text(video_id):
    api = YouTubeTranscriptApi(
        # proxy_config=WebshareProxyConfig(
        #     proxy_username="<proxy-username>",
        #     proxy_password="<proxy-password>",
        # )
    )
    transcript = api.fetch(video_id, languages=("en",))
    return " ".join(snippet.text for snippet in transcript)


def backfill_ids(max_videos=500):
    """One-off: match existing 'GDC - <title>.txt' files already sitting in
    RAW_DIR against the current channel listing by filename, to recover
    video ids for files that were downloaded before IDS_MANIFEST_PATH
    existed. A file whose video was retitled on YouTube since it was
    downloaded won't match anything in today's listing -- those are printed
    so they can be checked by hand instead of silently left unprotected."""
    print(f"Listing up to {max_videos} videos from {CHANNEL_URL} to backfill ids...")
    videos = list_channel_videos(max_videos)
    title_to_id = {f"GDC - {sanitize_filename(v['title'])}.txt": v["id"] for v in videos}

    scraped_ids = load_scraped_ids()
    matched, unmatched = 0, []
    for path in sorted(RAW_DIR.glob("GDC - *.txt")):
        video_id = title_to_id.get(path.name)
        if video_id:
            scraped_ids[video_id] = path.name[len("GDC - "):-len(".txt")]
            matched += 1
        else:
            unmatched.append(path.name)

    save_scraped_ids(scraped_ids)
    print(f"Backfilled {matched} ids into {IDS_MANIFEST_PATH}.")
    if unmatched:
        print(f"\n{len(unmatched)} existing file(s) didn't match any video in today's channel listing "
              f"(likely retitled on YouTube since they were downloaded) -- not protected by the id-based "
              f"dedup check yet, only by the old filename check:")
        for name in unmatched:
            print(f"  {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=150, help="Max videos to consider from the channel listing")
    parser.add_argument("--keywords", type=str, default="",
                         help="Comma-separated keywords; only titles containing one of these are kept "
                              "(case-insensitive). Empty = no filter.")
    parser.add_argument("--backfill-ids", action="store_true",
                         help="Don't scrape -- just match existing RAW_DIR files against the current channel "
                              "listing and record their video ids in the manifest, then exit.")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.backfill_ids:
        backfill_ids(args.max)
        return

    keywords = [k.strip().lower() for k in args.keywords.split(",") if k.strip()]

    print(f"Listing up to {args.max} videos from {CHANNEL_URL} ...")
    videos = list_channel_videos(args.max)
    if keywords:
        videos = [v for v in videos if any(k in v["title"].lower() for k in keywords)]
    print(f"{len(videos)} videos to process after filtering.")

    scraped_ids = load_scraped_ids()
    content_hashes = load_existing_content_hashes()
    needs_review_dir = RAW_DIR / "_needs_review"
    written, skipped_existing, skipped_no_transcript, flagged, deduped = 0, 0, 0, 0, 0

    for v in videos:
        out_path = RAW_DIR / f"GDC - {sanitize_filename(v['title'])}.txt"
        if v["id"] in scraped_ids or out_path.exists():
            skipped_existing += 1
            continue

        for attempt in range(1, MAX_BLOCK_RETRIES + 2):
            try:
                text = fetch_transcript_text(v["id"])
                break
            except (TranscriptsDisabled, NoTranscriptFound):
                print(f"  [skip: no transcript] {v['title']}")
                text = None
                break
            except RequestBlocked:
                if attempt > MAX_BLOCK_RETRIES:
                    print(f"\nStill blocked after {MAX_BLOCK_RETRIES} cooldowns -- stopping here instead of "
                          f"burning through the rest of the list on a dead IP. Re-run later to pick up where "
                          f"this left off (already-downloaded files are skipped automatically).")
                    print(f"\nDone. Written: {written} (flagged for review: {flagged}, routed to quarantine as duplicate content: {deduped}), "
                          f"already existed: {skipped_existing}, no transcript/error: {skipped_no_transcript}")
                    return
                print(f"  [blocked, attempt {attempt}/{MAX_BLOCK_RETRIES}] cooling down "
                      f"{BLOCK_COOLDOWN_SECONDS}s before retrying...")
                time.sleep(BLOCK_COOLDOWN_SECONDS)
            except Exception as e:
                print(f"  [error] {v['title']}: {e}")
                text = None
                break

        if not text:
            skipped_no_transcript += 1
            continue

        text_hash = _hash_text(text)
        existing = content_hashes.get(text_hash)

        if existing:
            # Same transcript already on disk under a different video id/title
            # (see QUARANTINE_DIR comment) -- goes straight to quarantine
            # rather than duplicating it into the live corpus.
            QUARANTINE_DIR.mkdir(exist_ok=True)
            save_path = QUARANTINE_DIR / out_path.name
            save_path.write_text(text)
            deduped += 1
            print(f"  [DUPLICATE content, saved to quarantine_dupes/ -- matches {existing.name}] {v['title']}")
        elif title_matches_content(v["title"], text):
            save_path = out_path
            out_path.write_text(text)
            print(f"  [saved] {v['title']}")
        else:
            # Saved under _needs_review/ (a subfolder text_processor.py never
            # scans, since it only reads RAW_DIR's direct children) instead
            # of the live corpus -- see IDS_MANIFEST_PATH comment for why.
            needs_review_dir.mkdir(exist_ok=True)
            save_path = needs_review_dir / out_path.name
            save_path.write_text(text)
            flagged += 1
            print(f"  [FLAGGED for review -- transcript doesn't look like it matches the title] {v['title']}")

        content_hashes[text_hash] = save_path
        scraped_ids[v["id"]] = v["title"]
        save_scraped_ids(scraped_ids)
        written += 1
        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nDone. Written: {written} (flagged for review: {flagged}, routed to quarantine as duplicate content: {deduped}), "
          f"already existed: {skipped_existing}, no transcript/error: {skipped_no_transcript}")


if __name__ == "__main__":
    main()
