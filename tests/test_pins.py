"""The pins file format (scripts/*.pins): one KEY="value" per line with a plain literal value. The download scripts
source the file and ``read_pins`` parses it, so anything the two could read differently is rejected."""

from __future__ import annotations

import pytest

from detmaso.data.ingest.pins import read_pins


def test_reads_quoted_values_and_skips_comments_and_blank_lines(tmp_path):
    path = tmp_path / "x.pins"
    path.write_text('# a comment\n\nA_COMMIT="d3b7036"\nSP500_FILE="S&P 500 Components & Changes (Updated).csv"\n\n')
    assert read_pins(path) == {"A_COMMIT": "d3b7036", "SP500_FILE": "S&P 500 Components & Changes (Updated).csv"}


@pytest.mark.parametrize(
    "text",
    [
        "A=d3b7036\n",  # unquoted
        "A='d3b7036'\n",  # single quotes
        'A=""\n',  # empty
        'A="$HOME/x"\n',  # bash would expand it
        'A="`id`"\n',
        'A="x\\y"\n',
        'a="x"\n',  # lower-case key
        'A="x"  # note\n',  # trailing comment
        ' A="x"\n',  # leading space
        'A="x" B="y"\n',  # two per line
        'A="x"\nA="y"\n',  # duplicate key
        "# only a comment\n",
    ],
)
def test_rejects_what_bash_and_python_could_read_differently(tmp_path, text):
    path = tmp_path / "x.pins"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_pins(path)
