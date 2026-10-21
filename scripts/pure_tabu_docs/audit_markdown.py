"""Check historical numbers and newly added links in the synchronized documents."""
import json
from pathlib import Path
import re

from sync_markdown import BEGIN, END, MAIN_BEGIN, MAIN_END, MAIN_ROW, OUT, ROOT, TEACHING_FILES


def historical_numbers(text):
    text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), "", text, flags=re.S)
    text = re.sub(re.escape(MAIN_BEGIN) + r".*?" + re.escape(MAIN_END), "", text, flags=re.S)
    text = re.sub(r"^.*" + re.escape(MAIN_ROW) + r"\n?", "", text, flags=re.M)
    text = re.sub(r"\]\([^\n)]*\)", "]", text)
    text = re.sub(r"outputs/experiments/[^\s`，；)]+\.png", "IMAGE_PATH", text)
    return re.findall(r"\d+(?:\.\d+)?", text)


def main():
    names = json.loads((OUT / "markdown_updated.json").read_text(encoding="utf-8"))
    mismatches, links = [], []
    for name in names:
        current = ROOT / name
        old = OUT / "document_backups" / name
        text = current.read_text(encoding="utf-8-sig")
        # The teaching pages are being revised concurrently by their owning chat.
        # Their algorithm counts/navigation are not historical experiment values.
        if name not in TEACHING_FILES and historical_numbers(text) != historical_numbers(old.read_text(encoding="utf-8-sig")):
            mismatches.append(name)
        block = re.search(re.escape(BEGIN) + r"(.*?)" + re.escape(END), text, flags=re.S).group(1)
        for target in re.findall(r"\]\(([^)]+)\)", block):
            if not (current.parent / target).resolve().exists():
                links.append({"file": name, "target": target})
    print(json.dumps({"documents": len(names), "historical_number_changes": mismatches, "broken_new_links": links}, ensure_ascii=True))
    if mismatches or links:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
