from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rich.text import Text

RULE_WIDTH = 88

NARROW_COLUMN_WIDTH = 24

TITLE_STYLE = "bold"
FIELD_KEY_STYLE = "dim"
SECTION_STYLE = "bold cyan"
TABLE_HEADER_STYLE = "bold magenta"


def styled(text: str, style: str) -> Text:
    """A table cell that is coloured on the terminal and plain in the saved report."""
    from rich.text import Text

    return Text(text, style=style)


def plain(cell: Any) -> str:
    """The visible characters of a cell, ignoring any styling."""
    return cell.plain if hasattr(cell, "plain") else str(cell)


class Report:
    """Accumulates report content, then prints it in colour and optionally saves it as plain
    text."""

    def __init__(self, title: str, fields: Sequence[tuple[str, str]] | None = None) -> None:
        self.title = title
        self.fields = list(fields or [])
        self._blocks: list[tuple] = []

    # --- building ---------------------------------------------------------------------------

    def section(self, name: str) -> Report:
        """Starts a new named section."""
        self._blocks.append(("section", name))
        return self

    def kv(self, key: str, value: object) -> Report:
        """Adds one aligned `key : value` line."""
        self._blocks.append(("kv", key, value))
        return self

    def note(self, text: str | Text) -> Report:
        """Adds a free-form line, e.g. an interpretation or a warning."""
        self._blocks.append(("note", text))
        return self

    def blank(self) -> Report:
        self._blocks.append(("blank",))
        return self

    def raw(self, text: str) -> Report:
        """Adds a line that must survive verbatim -- a command meant to be copied and pasted."""
        self._blocks.append(("raw", text))
        return self

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[object]]) -> Report:
        """Adds a table. Cells may be plain values or `styled()` text.

        Empty `rows` adds a placeholder instead of a bare header.
        """
        if not rows:
            return self.note("(no rows)")

        widths = {len(row) for row in rows}
        if widths != {len(headers)}:
            # The previous renderer silently dropped extra cells, which hid a mismatched table for a
            # while. Fail loudly instead.
            raise ValueError(
                f"Table has {len(headers)} headers {list(headers)} but rows of width "
                f"{sorted(widths)}."
            )

        self._blocks.append(("table", list(headers), [list(row) for row in rows]))
        return self

    # --- plain-text output ------------------------------------------------------------------

    def render(self) -> str:
        """The whole report as plain text, for saving."""
        head = ["=" * RULE_WIDTH, self.title]
        head.extend(f"  {key:<28} {value}" for key, value in self.fields)
        head.append("=" * RULE_WIDTH)
        return "\n".join([*head, "", *self._render_blocks()]) + "\n"

    def _render_blocks(self) -> list[str]:
        lines: list[str] = []
        for block in self._blocks:
            kind = block[0]
            if kind == "section":
                if lines:
                    lines.append("")
                lines.append(f"## {block[1]}")
            elif kind == "kv":
                lines.append(f"  {block[1]:<28} {block[2]}")
            elif kind in ("note", "raw"):
                lines.append(f"  {plain(block[1])}")
            elif kind == "blank":
                lines.append("")
            elif kind == "table":
                lines.extend(_plain_table(block[1], block[2]))
        return lines

    # --- terminal output --------------------------------------------------------------------

    def print(self) -> None:
        """Prints the report to the terminal, in colour."""
        from rich.console import Console
        from rich.rule import Rule
        from rich.text import Text

        # highlight=False: the default would recolour every number and path in the report at random.
        console = Console(highlight=False)

        console.print(Rule(Text(self.title, style=TITLE_STYLE), characters="="))
        for key, value in self.fields:
            console.print(Text(f"  {key:<28} ", style=FIELD_KEY_STYLE), Text(str(value)), sep="")
        console.print()

        for block in self._blocks:
            kind = block[0]
            if kind == "section":
                console.print()
                console.print(Text(block[1], style=SECTION_STYLE))
            elif kind == "kv":
                console.print(
                    Text(f"  {block[1]:<28} ", style=FIELD_KEY_STYLE), Text(str(block[2])), sep=""
                )
            elif kind == "note":
                console.print(Text("  "), block[1], sep="")
            elif kind == "raw":
                # soft_wrap so rich never inserts a newline mid-command; the terminal may still fold
                # it visually, which leaves the copied text intact.
                console.print(Text(f"  {block[1]}"), soft_wrap=True)
            elif kind == "blank":
                console.print()
            elif kind == "table":
                console.print(_rich_table(block[1], block[2]))

    # --- output -----------------------------------------------------------------------------

    def emit(self, path: Path | None = None, *, save: bool = True) -> Path | None:
        """Prints the report, and writes it alongside the figure unless `save` is False.

        Returns the path written, or None if nothing was saved.
        """
        self.print()

        if not save or path is None:
            return None

        report_path = path.with_suffix(".txt")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(self.render(), encoding="utf-8")
        print(f"Report saved: {report_path}")
        return report_path


def _plain_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    """Column-aligned ASCII table."""
    cells = [[plain(cell) for cell in row] for row in rows]
    widths = [
        max(len(str(headers[i])), *(len(row[i]) for row in cells)) for i in range(len(headers))
    ]

    lines = [
        "  | " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |",
        "  |-" + "-|-".join("-" * width for width in widths) + "-|",
    ]
    lines.extend(
        "  | " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in cells
    )
    return lines


def _rich_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]):
    """The same table as a `rich.table.Table`, mirroring `policy/experiment.py`'s conventions."""
    from rich.table import Table

    table = Table(show_header=True, header_style=TABLE_HEADER_STYLE)
    for index, header in enumerate(headers):
        values = [plain(row[index]) for row in rows]
        widest = max(len(str(header)), *(len(value) for value in values))
        table.add_column(
            str(header),
            # Right-justify columns whose every value reads as a number or a size.
            justify="right" if _is_numeric(values) else "left",
            # Narrow columns must never be shrunk: truncating a timestamp or a status destroys the
            # information. Only genuinely long cells are allowed to give way, and they wrap rather
            # than ellipsise so nothing is silently lost.
            no_wrap=widest <= NARROW_COLUMN_WIDTH,
            overflow="fold",
        )
    for row in rows:
        table.add_row(*(cell if hasattr(cell, "plain") else str(cell) for cell in row))
    return table


def _is_numeric(values: Sequence[str]) -> bool:
    """Whether a column holds only numbers, byte sizes (`4.4G`), or `-` placeholders."""
    return bool(values) and all(
        value == "-" or value.rstrip("BKMG").replace(".", "", 1).isdigit() for value in values
    )
