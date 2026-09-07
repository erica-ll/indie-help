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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=150, help="Max videos to consider from the channel listing")
    parser.add_argument("--keywords", type=str, default="",
                         help="Comma-separated keywords; only titles containing one of these are kept "
                              "(case-insensitive). Empty = no filter.")
    args = parser.parse_args()

    keywords = [k.strip().lower() for k in args.keywords.split(",") if k.strip()]

    print(f"Listing up to {args.max} videos from {CHANNEL_URL} ...")
    videos = list_channel_videos(args.max)
    if keywords:
        videos = [v for v in videos if any(k in v["title"].lower() for k in keywords)]
    print(f"{len(videos)} videos to process after filtering.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    written, skipped_existing, skipped_no_transcript = 0, 0, 0

    for v in videos:
        out_path = RAW_DIR / f"GDC - {sanitize_filename(v['title'])}.txt"
        if out_path.exists():
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
                    print(f"\nDone. Written: {written}, already existed: {skipped_existing}, "
                          f"no transcript/error: {skipped_no_transcript}")
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

        out_path.write_text(text)
        written += 1
        print(f"  [saved] {v['title']}")
        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nDone. Written: {written}, already existed: {skipped_existing}, no transcript/error: {skipped_no_transcript}")


if __name__ == "__main__":
    main()
