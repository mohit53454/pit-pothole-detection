"""Download RDD2022/India.zip (530 MB) out of the 13 GB RDD2022 zip remotely.

The figshare release (article 21431547) is one 13 GB zip containing per-country
zips. `remotezip` fetches only the central directory + the India member via
HTTP range requests, so we download ~530 MB instead of 13 GB.
"""

from __future__ import annotations

import time
from pathlib import Path

from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "RDD2022" / "India.zip"
URL = "https://ndownloader.figshare.com/files/38030910"
MEMBER = "RDD2022/India.zip"


def main() -> int:
    if DEST.exists():
        print(f"already downloaded: {DEST}")
        return 0
    DEST.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with RemoteZip(URL) as z:
        z.extract(MEMBER, path=ROOT / "data")
    print(f"downloaded in {time.time() - t0:.0f}s -> {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
