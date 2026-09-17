"""Writes web/sheets.json listing which thumbnail sheets exist.

Without it the viewer has to probe each sheet URL and eat a 404 for every sheet that has not
been generated yet. Run this after build_previews.py and before committing.
"""

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THUMBS = os.path.join(ROOT, "web", "thumbnails")
OUT = os.path.join(ROOT, "web", "sheets.json")

ids = sorted(int(m.group(1))
             for f in os.listdir(THUMBS) if (m := re.fullmatch(r"sheet_(\d+)\.webp", f)))
with open(OUT, "w") as f:
    json.dump({"available": ids}, f, separators=(",", ":"))
print(f"{len(ids)} sheets -> {os.path.relpath(OUT, ROOT)}")
