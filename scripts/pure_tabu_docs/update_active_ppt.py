"""Insert the pure-tabu row into the already open codex presentation."""

import json
from pathlib import Path

import win32com.client


ROOT = Path(__file__).resolve().parents[2]
PPT = (ROOT / "docs/Claude维护/导师汇报/codex版/02_导师汇报.pptx").resolve()
SUMMARY = ROOT / "outputs/experiments/s2_pure_tabu_20261021/presentation_summary.json"


def rgb(red, green, blue):
    return red + 256 * green + 65536 * blue


def main():
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    pure = next(row for row in data["rows"] if row["set"] == "conf_d432_n150" and row["method"] == "PTS32")
    app = win32com.client.GetActiveObject("PowerPoint.Application")
    presentation = next(item for item in app.Presentations if Path(item.FullName).resolve() == PPT)
    slide = presentation.Slides(15)
    table = next(shape.Table for shape in slide.Shapes
                 if shape.HasTable and shape.Table.Cell(1, 1).Shape.TextFrame.TextRange.Text == "方法")
    labels = [table.Cell(index, 1).Shape.TextFrame.TextRange.Text for index in range(1, table.Rows.Count + 1)]
    if any("纯禁忌搜索32次" in label for label in labels):
        print("unchanged: pure-tabu row already in the main table")
        return
    assert labels[4] == "容量感知贪心"
    table.Rows.Add(6)
    values = ["纯禁忌搜索32次（随机初解）", f"{pure['after_W'] * 100:.1f}%",
              f"{pure['after_aircraft_worst10'] * 100:.1f}%", f"{pure['after_SGM']:.3f}"]
    for column, value in enumerate(values, start=1):
        cell_shape = table.Cell(6, column).Shape
        cell_shape.TextFrame.TextRange.Text = value
        cell_shape.Fill.Solid()
        cell_shape.Fill.ForeColor.RGB = rgb(225, 244, 239)
        font = cell_shape.TextFrame.TextRange.Font
        font.Name = "Microsoft YaHei"
        font.Size = 14
        font.Bold = True
        font.Color.RGB = rgb(13, 95, 77)
    table.Rows(1).Height = 35.28
    for index in range(2, table.Rows.Count + 1):
        table.Rows(index).Height = 36
    presentation.Save()
    print("saved: slide 15 pure-tabu row in " + str(PPT))


if __name__ == "__main__":
    main()
