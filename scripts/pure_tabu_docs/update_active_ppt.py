"""Insert the pure-tabu row into the already open codex presentation."""

import json
from pathlib import Path

import win32com.client


ROOT = Path(__file__).resolve().parents[2]
PPT = (ROOT / "docs/Claude维护/导师汇报/codex版/02_导师汇报.pptx").resolve()
SUMMARY = ROOT / "outputs/experiments/s2_pure_tabu_20261021/presentation_summary.json"


def main():
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    pure = next(row for row in data["rows"] if row["set"] == "conf_d432_n150" and row["method"] == "PTS32")
    app = win32com.client.GetActiveObject("PowerPoint.Application")
    presentation = next(item for item in app.Presentations if Path(item.FullName).resolve() == PPT)
    slide = presentation.Slides(15)
    table = next(shape.Table for shape in slide.Shapes
                 if shape.HasTable and shape.Table.Cell(1, 1).Shape.TextFrame.TextRange.Text == "方法")
    labels = [table.Cell(index, 1).Shape.TextFrame.TextRange.Text for index in range(1, table.Rows.Count + 1)]
    if not any("纯禁忌搜索32次" in label for label in labels):
        assert labels[4] == "容量感知贪心"
        table.Rows.Add(6)
    values = ["纯禁忌搜索32次（随机初解）", f"{pure['after_W'] * 100:.1f}%",
              f"{pure['after_aircraft_worst10'] * 100:.1f}%", f"{pure['after_SGM']:.3f}"]
    values[0] = "纯禁忌搜索32次"
    changed = False
    for column, value in enumerate(values, start=1):
        cell_shape = table.Cell(6, column).Shape
        template = table.Cell(4, column).Shape
        if cell_shape.TextFrame.TextRange.Text != value:
            cell_shape.TextFrame.TextRange.Text = value
            changed = True
        if cell_shape.Fill.ForeColor.RGB != template.Fill.ForeColor.RGB:
            changed = True
        cell_shape.Fill.Solid()
        cell_shape.Fill.ForeColor.RGB = template.Fill.ForeColor.RGB
        font = cell_shape.TextFrame.TextRange.Font
        source_font = template.TextFrame.TextRange.Font
        far_east = "宋体" if column == 1 else source_font.Name
        if (font.Size != source_font.Size or font.Bold != source_font.Bold or
                font.Color.RGB != source_font.Color.RGB or font.NameFarEast != far_east):
            changed = True
        font.Name = "" if column == 1 else source_font.Name
        font.NameFarEast = far_east
        font.Size = source_font.Size
        font.Bold = source_font.Bold
        font.Color.RGB = source_font.Color.RGB
    first_color = table.Cell(2, 1).Shape.Fill.ForeColor.RGB
    second_color = table.Cell(3, 1).Shape.Fill.ForeColor.RGB
    for index in range(2, table.Rows.Count + 1):
        expected = first_color if index % 2 == 0 else second_color
        for column in range(1, table.Columns.Count + 1):
            cell = table.Cell(index, column).Shape
            if cell.Fill.ForeColor.RGB != expected:
                cell.Fill.Solid()
                cell.Fill.ForeColor.RGB = expected
                changed = True
    table.Rows(1).Height = 35.28
    for index in range(2, table.Rows.Count + 1):
        table.Rows(index).Height = 36
    if changed:
        presentation.Save()
        print("saved: slide 15 pure-tabu row in " + str(PPT))
    else:
        print("unchanged: slide 15 pure-tabu row matches the table style")


if __name__ == "__main__":
    main()
