"""Refresh codex-version PDFs from the synchronized Office files."""

from pathlib import Path
import sys

import pythoncom
import win32com.client


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "docs/Claude维护/导师汇报/codex版"
WORD_STEMS = ("01_导师汇报材料", "03_讲解提纲", "05_中期研究进展同步稿")


def export_word():
    application = win32com.client.DispatchEx("Word.Application")
    application.Visible = False
    application.DisplayAlerts = 0
    try:
        for stem in WORD_STEMS:
            source = FOLDER / f"{stem}.docx"
            document = application.Documents.Open(str(source), ReadOnly=True, AddToRecentFiles=False, Visible=False)
            try:
                document.ExportAsFixedFormat(str(FOLDER / f"{stem}.pdf"), 17)
            finally:
                document.Close(False)
    finally:
        application.Quit()


def export_powerpoint():
    application = win32com.client.DispatchEx("PowerPoint.Application")
    source = FOLDER / "02_导师汇报.pptx"
    presentation = application.Presentations.Open(str(source), ReadOnly=True, Untitled=False, WithWindow=False)
    try:
        presentation.SaveAs(str(FOLDER / "02_导师汇报.pdf"), 32)
    finally:
        presentation.Close()


def main():
    pythoncom.CoInitialize()
    try:
        if "--ppt-only" not in sys.argv:
            export_word()
        export_powerpoint()
    finally:
        pythoncom.CoUninitialize()


if __name__ == "__main__":
    main()
