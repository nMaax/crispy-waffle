"""Tests for the report renderer used by every script under `scripts/`.

The point of the two renderers is that colour never reaches the saved `.txt`, and that styling never
has to be embedded in the report text as markup -- reports legitimately contain literal square
brackets that markup parsing would eat.
"""

import pytest

from scripts.utils.report import Report, plain, styled


class TestPlainRendering:
    def test_header_carries_title_and_fields(self):
        text = Report("My report", [("ckpt", "logs/x/last.ckpt"), ("seed", "42")]).render()
        assert "My report" in text
        assert "  ckpt                         logs/x/last.ckpt" in text
        assert text.startswith("=" * 88)

    def test_blocks_render_in_order(self):
        text = (
            Report("t")
            .section("First")
            .kv("key", 1)
            .note("a note")
            .blank()
            .section("Second")
            .render()
        )
        assert text.index("## First") < text.index("key") < text.index("## Second")

    def test_table_is_column_aligned(self):
        text = Report("t").table(["a", "bbbb"], [["1", "2"], ["333", "4"]]).render()
        lines = [line for line in text.splitlines() if line.startswith("  |")]
        assert len(lines) == 4  # header, divider, two rows
        assert len({len(line) for line in lines}) == 1, "rows are not the same width"

    def test_empty_table_degrades_to_a_note(self):
        assert "(no rows)" in Report("t").table(["a"], []).render()


class TestTableShape:
    def test_rejects_rows_that_do_not_match_the_headers(self):
        """The old renderer silently dropped extra cells, which hid a mismatched table for a
        while."""
        with pytest.raises(ValueError, match="2 headers"):
            Report("t").table(["a", "b"], [["1", "2", "3"]])

    def test_rejects_rows_of_differing_width(self):
        with pytest.raises(ValueError):
            Report("t").table(["a", "b"], [["1", "2"], ["1"]])


class TestRawLines:
    COMMAND = 'HfApi().delete_folder(path_in_repo="logs/a/runs/b/c", repo_id="x/y")'

    def test_appears_in_the_saved_report(self):
        assert self.COMMAND in Report("t").raw(self.COMMAND).render()

    def test_is_not_rewrapped_on_the_terminal(self, capsys):
        """A wrapped command cannot be pasted, so rich must not insert a newline into it."""
        Report("t").raw(self.COMMAND).print()
        assert self.COMMAND in capsys.readouterr().out


class TestStyledCells:
    def test_style_does_not_reach_the_plain_render(self):
        text = Report("t").table(["state"], [[styled("completed", "green")]]).render()
        assert "completed" in text
        assert "green" not in text
        assert "\033" not in text, "an escape code leaked into the saved report"

    def test_style_does_not_inflate_column_width(self):
        styled_report = Report("t").table(["s"], [[styled("completed", "green")]]).render()
        plain_report = Report("t").table(["s"], [["completed"]]).render()
        assert styled_report == plain_report

    def test_styled_note_renders_as_plain_text(self):
        assert "  # superseded" in Report("t").note(styled("# superseded", "red")).render()

    def test_plain_helper_handles_both_kinds(self):
        assert plain(styled("x", "red")) == "x"
        assert plain(7) == "7"


class TestMarkupHazard:
    """Reports contain literal `[...]`; it must survive both renderers untouched.

    `analyze_goal_signal_convergence.py` prints a LayerNorm band as `[6.216, 9.290]`, and
    `analyze_dataset_biases.py` embeds `np.array2string` output. Rendering those through rich with
    markup enabled would silently delete them.
    """

    BRACKETS = "[6.216, 9.290]"

    def test_survives_the_plain_render(self):
        assert self.BRACKETS in Report("t").note(f"band {self.BRACKETS}").render()

    def test_survives_a_table_cell(self):
        assert self.BRACKETS in Report("t").table(["band"], [[self.BRACKETS]]).render()

    def test_survives_the_terminal_render(self, capsys):
        Report("t").note(f"band {self.BRACKETS}").print()
        assert self.BRACKETS in capsys.readouterr().out


class TestEmit:
    def test_saves_next_to_the_given_path_with_a_txt_suffix(self, tmp_path):
        written = Report("t").note("hello").emit(tmp_path / "figures" / "plot.png")
        assert written == tmp_path / "figures" / "plot.txt"
        assert "hello" in written.read_text()

    def test_writes_nothing_when_save_is_false(self, tmp_path):
        target = tmp_path / "plot.png"
        assert Report("t").note("hello").emit(target, save=False) is None
        assert not target.with_suffix(".txt").exists()

    def test_writes_nothing_without_a_path(self):
        assert Report("t").note("hello").emit(None, save=True) is None

    def test_saved_file_has_no_escape_codes(self, tmp_path):
        written = Report("t").table(["s"], [[styled("completed", "green")]]).emit(tmp_path / "p.png")
        assert written is not None
        assert "\033" not in written.read_text()
