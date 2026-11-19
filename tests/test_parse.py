"""The verdict parser: one word, PASS or FAIL, taken from the visible answer only."""

from labjudge.judge_mlx import parse


def test_plain_words():
    assert parse("PASS") == "PASS"
    assert parse(" fail.") == "FAIL"


def test_anything_else_is_unparsed():
    assert parse("PAS") == "unparsed"
    assert parse(" REFUSED") == "unparsed"
    assert parse("") == "unparsed"


def test_thinking_text_is_not_the_answer():
    reply = "I must answer PASS or FAIL. The plate is off the tray.</think>FAIL"
    assert parse(reply, thinking=True) == "FAIL"


def test_unfinished_thinking_has_no_answer():
    reply = "I must answer PASS or FAIL. Let me look at the top view"
    assert parse(reply, thinking=True) == "unparsed"


def test_first_word_wins_in_the_visible_answer():
    assert parse("</think>PASS, not FAIL") == "PASS"
