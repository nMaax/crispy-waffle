from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

RULE_WIDTH = 88


class Report:
    """Accumulates report content, then prints and optionally saves it."""

    def __init__(self, title: str, fields: Sequence[tuple[str, str]] | None = None) -> None:
        self.title = title
        self.fields = list(fields or [])
        self._lines: list[str] = []

    # --- building ---------------------------------------------------------------------------

    def section(self, name: str) -> Report:
        """Starts a new named section."""
        if self._lines:
            self._lines.append("")
        self._lines.append(f"## {name}")
        return self

    def kv(self, key: str, value: object) -> Report:
        """Adds one aligned `key : value` line."""
        self._lines.append(f"  {key:<28} {value}")
        return self

    def note(self, text: str) -> Report:
        """Adds a free-form line, e.g. an interpretation or a warning."""
        self._lines.append(f"  {text}")
        return self

    def blank(self) -> Report:
        self._lines.append("")
        return self

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[object]]) -> Report:
        """Adds a column-aligned table.

        Empty `rows` adds a placeholder instead of a bare header.
        """
        if not rows:
            return self.note("(no rows)")

        cells = [[str(c) for c in row] for row in rows]
        widths = [
            max(len(str(headers[i])), *(len(row[i]) for row in cells)) for i in range(len(headers))
        ]

        header_line = "  | " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
        divider = "  |-" + "-|-".join("-" * w for w in widths)
        self._lines.append(header_line + " |")
        self._lines.append(divider + "-|")
        for row in cells:
            self._lines.append(
                "  | " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
            )
        return self

    # --- output -----------------------------------------------------------------------------

    def render(self) -> str:
        """Returns the full report as a string."""
        head = ["=" * RULE_WIDTH, self.title]
        if self.fields:
            head.extend(f"  {key:<28} {value}" for key, value in self.fields)
        head.append("=" * RULE_WIDTH)
        return "\n".join([*head, "", *self._lines]) + "\n"

    def emit(self, path: Path | None = None, *, save: bool = True) -> Path | None:
        """Prints the report, and writes it alongside the figure unless `save` is False.

        Returns the path written, or None if nothing was saved.
        """
        text = self.render()
        print(text)

        if not save or path is None:
            return None

        report_path = path.with_suffix(".txt")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text, encoding="utf-8")
        print(f"Report saved: {report_path}")
        return report_path
