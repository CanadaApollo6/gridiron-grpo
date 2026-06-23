"""Comprehensive unit tests for src/rewards/verifiers.py.

These reward functions are the contract the GRPO trainer optimizes against AND
the contract the eval scores against. A silent regression here corrupts both
training signal and every reported number, so the parsing, normalization,
per-answer-type checking, graded partial credit, and the two TRL-facing reward
entry points are all pinned here.

Run: `pytest`  (src path + config come from pyproject.toml / conftest.py)
"""

import pytest

from rewards.verifiers import (
    _check,
    _name_match,
    _norm_name,
    _norm_set,
    _numeric_partial,
    _parse_number,
    _to_text,
    correctness_reward,
    correctness_reward_graded,
    format_reward,
)


def _ans(x) -> str:
    """A well-formed <think>/<answer> completion carrying answer `x`."""
    return f"<think>some reasoning</think><answer>{x}</answer>"


# --------------------------------------------------------------------------- #
# _to_text — normalize a string OR a chat-message-list completion to plain text
# --------------------------------------------------------------------------- #
class TestToText:
    def test_plain_string_passthrough(self):
        assert _to_text("hello world") == "hello world"

    def test_chat_list_joins_content(self):
        msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        assert _to_text(msgs) == "a b"

    def test_chat_list_missing_content_is_empty_str(self):
        assert _to_text([{"role": "assistant"}]) == ""

    def test_chat_list_skips_non_dict_items(self):
        assert _to_text([{"role": "x", "content": "ok"}, "stray", 7]) == "ok"

    def test_non_str_non_list_is_stringified(self):
        assert _to_text(42) == "42"
        assert _to_text(None) == "None"


# --------------------------------------------------------------------------- #
# _parse_number — pull the first signed int/float out of free text
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "s,expected",
    [
        ("137", 137.0),
        ("-4", -4.0),  # real box scores have negative rush yards
        ("12.5", 12.5),
        ("1,234", 1234.0),  # commas stripped before matching
        ("3,515.23", 3515.23),
        ("the total is 42 yards", 42.0),
        ("007", 7.0),
        ("-12.0", -12.0),
        ("3.14.15", 3.14),  # first valid number only
    ],
)
def test_parse_number_extracts(s, expected):
    assert _parse_number(s) == expected


@pytest.mark.parametrize("s", ["", "no digits here", None, "   ", "abc."])
def test_parse_number_returns_none(s):
    assert _parse_number(s) is None


# --------------------------------------------------------------------------- #
# _norm_name — lowercase, drop dots, collapse whitespace
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "s,expected",
    [
        ("A. Smith", "a smith"),
        ("  E. Thomas  ", "e thomas"),
        ("St. Brown", "st brown"),
        ("J.  Doe", "j doe"),  # multiple spaces collapse to one
        ("ALL CAPS", "all caps"),
        (None, ""),
        ("", ""),
    ],
)
def test_norm_name(s, expected):
    assert _norm_name(s) == expected


# --------------------------------------------------------------------------- #
# _norm_set — order-insensitive set; split on , / newline / ; ; drop none+empties
# --------------------------------------------------------------------------- #
class TestNormSet:
    def test_basic(self):
        assert _norm_set("A. Smith, J. Doe") == {"a smith", "j doe"}

    def test_order_insensitive(self):
        assert _norm_set("G. Martin, J. Garcia") == _norm_set("J. Garcia, G. Martin")

    @pytest.mark.parametrize("sep", [",", "\n", ";"])
    def test_separators(self, sep):
        assert _norm_set(f"A{sep}B{sep}C") == {"a", "b", "c"}

    def test_discards_empties(self):
        assert _norm_set("A,,B,") == {"a", "b"}

    def test_discards_none_token(self):
        assert _norm_set("none") == set()
        assert _norm_set("A, none, B") == {"a", "b"}  # 'none' dropped even when mixed in

    @pytest.mark.parametrize("s", ["", None])
    def test_empty_inputs(self, s):
        assert _norm_set(s) == set()


# --------------------------------------------------------------------------- #
# _name_match — exact OR last-name-only (the documented leniency)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "pred,gt",
    [
        ("E. Thomas", "E. Thomas"),  # exact
        ("e. thomas", "E. Thomas"),  # case-insensitive
        ("Thomas", "E. Thomas"),  # last-name only
        ("E. THOMAS", "e. thomas"),
    ],
)
def test_name_match_true(pred, gt):
    assert _name_match(pred, gt) is True


@pytest.mark.parametrize(
    "pred,gt",
    [
        ("E. Smith", "E. Thomas"),  # different last name
        ("", "E. Thomas"),  # empty pred
        (None, "E. Thomas"),  # None pred
        ("E. Thomas", ""),  # empty gt
    ],
)
def test_name_match_false(pred, gt):
    assert _name_match(pred, gt) is False


def test_name_match_last_name_leniency_is_intentional():
    # Different first initials, same last name -> accepted by design. The
    # generators enforce unique last names within a box so this stays well-posed.
    assert _name_match("J. Thomas", "E. Thomas") is True


# --------------------------------------------------------------------------- #
# _check — the per-answer-type dispatcher
# --------------------------------------------------------------------------- #
class TestCheckNumeric:
    @pytest.mark.parametrize("pred,gt", [("137", "137"), ("137.0", "137"), ("1,234", "1234")])
    def test_correct(self, pred, gt):
        assert _check(pred, gt, "numeric") is True

    @pytest.mark.parametrize("pred,gt", [("138", "137"), ("foo", "137")])
    def test_wrong(self, pred, gt):
        assert _check(pred, gt, "numeric") is False

    def test_tolerance_is_tight(self):
        assert _check("137.0000001", "137", "numeric") is True  # within 1e-6
        assert _check("137.01", "137", "numeric") is False  # outside 1e-6


class TestCheckName:
    def test_exact(self):
        assert _check("E. Thomas", "E. Thomas", "name") is True

    def test_last_name(self):
        assert _check("Thomas", "E. Thomas", "name") is True

    def test_wrong(self):
        assert _check("E. Smith", "E. Thomas", "name") is False


class TestCheckSet:
    def test_unordered_equal(self):
        assert _check("A. Smith, J. Doe", "J. Doe, A. Smith", "set") is True

    def test_none_equals_none(self):
        assert _check("none", "none", "set") is True

    def test_subset_is_wrong(self):
        assert _check("A. Smith", "A. Smith, J. Doe", "set") is False

    def test_superset_is_wrong(self):
        assert _check("A, B, C", "A, B", "set") is False


class TestCheckDecision:
    @pytest.mark.parametrize(
        "pred,gt",
        [
            ("TD", "TD"),
            ("touchdown", "TD"),
            ("the answer is TD", "TD"),  # td/fg match on substring
            ("FG", "FG"),
            ("field goal", "FG"),
            ("OVER", "OVER"),  # other decisions: exact token
            ("UNDER", "UNDER"),
        ],
    )
    def test_correct(self, pred, gt):
        assert _check(pred, gt, "decision") is True

    @pytest.mark.parametrize(
        "pred,gt",
        [
            ("FG", "TD"),
            ("TD", "FG"),
            ("UNDER", "OVER"),
            ("OVER", "UNDER"),
            ("OVER budget", "OVER"),  # non-td/fg decisions need an exact token
        ],
    )
    def test_wrong(self, pred, gt):
        assert _check(pred, gt, "decision") is False


def test_check_unknown_answer_type_is_false():
    assert _check("x", "x", "mystery_type") is False


@pytest.mark.parametrize("answer_type", ["numeric", "name", "set", "decision"])
def test_check_none_pred_is_false_for_every_type(answer_type):
    assert _check(None, "x", answer_type) is False


# --------------------------------------------------------------------------- #
# _numeric_partial — graded credit (training-only densification, eval stays strict)
# --------------------------------------------------------------------------- #
class TestNumericPartial:
    def test_zero_error_is_full_cap(self):
        # raw helper caps at 0.5; the graded reward returns 1.0 for exact via
        # _check before this is ever reached.
        assert _numeric_partial("100", "100") == pytest.approx(0.5)

    def test_within_band_scales_linearly(self):
        assert _numeric_partial("105", "100") == pytest.approx(0.25)  # err 5, band 10
        assert _numeric_partial("100.5", "100") == pytest.approx(0.475)

    def test_at_or_beyond_band_is_zero(self):
        assert _numeric_partial("110", "100") == 0.0  # err == band edge
        assert _numeric_partial("200", "100") == 0.0

    def test_floor_band_of_one_near_zero(self):
        assert _numeric_partial("1", "0") == 0.0  # err 1 >= band 1 (floor)
        assert _numeric_partial("0.5", "0") == pytest.approx(0.25)

    @pytest.mark.parametrize("pred", ["foo", None])
    def test_unparseable_is_zero(self, pred):
        assert _numeric_partial(pred, "100") == 0.0


# --------------------------------------------------------------------------- #
# correctness_reward — the strict 0/1 reward TRL calls
# --------------------------------------------------------------------------- #
class TestCorrectnessReward:
    def test_correct_and_wrong_numeric(self):
        out = correctness_reward(
            None,
            [_ans("137"), _ans("999")],
            ground_truth=["137", "137"],
            answer_type=["numeric", "numeric"],
        )
        assert out == [1.0, 0.0]

    def test_accepts_chat_list_completions(self):
        comps = [[{"role": "assistant", "content": _ans("E. Thomas")}]]
        out = correctness_reward(None, comps, ground_truth=["E. Thomas"], answer_type=["name"])
        assert out == [1.0]

    def test_missing_answer_block_scores_zero(self):
        out = correctness_reward(
            None, ["just 137, no answer tag"], ground_truth=["137"], answer_type=["numeric"]
        )
        assert out == [0.0]

    def test_uses_last_answer_block(self):
        # extract_answer takes the LAST <answer> (a restated final answer wins)
        comp = "<answer>111</answer> wait, <answer>137</answer>"
        out = correctness_reward(None, [comp], ground_truth=["137"], answer_type=["numeric"])
        assert out == [1.0]

    def test_mixed_batch_all_answer_types(self):
        comps = [_ans("280"), _ans("Thomas"), _ans("A. Smith, B. Lee"), _ans("FG")]
        gt = ["280", "E. Thomas", "B. Lee, A. Smith", "FG"]
        at = ["numeric", "name", "set", "decision"]
        out = correctness_reward(None, comps, ground_truth=gt, answer_type=at)
        assert out == [1.0, 1.0, 1.0, 1.0]

    def test_returns_floats_of_matching_length(self):
        comps = [_ans("1"), _ans("2"), _ans("3")]
        out = correctness_reward(
            None,
            comps,
            ground_truth=["1", "2", "9"],
            answer_type=["numeric"] * 3,
        )
        assert out == [1.0, 1.0, 0.0]
        assert all(isinstance(x, float) for x in out)


# --------------------------------------------------------------------------- #
# correctness_reward_graded — partial credit on numeric ONLY
# --------------------------------------------------------------------------- #
class TestCorrectnessRewardGraded:
    def test_exact_numeric_is_one(self):
        out = correctness_reward_graded(
            None, [_ans("100")], ground_truth=["100"], answer_type=["numeric"]
        )
        assert out == [1.0]

    def test_close_numeric_is_partial(self):
        out = correctness_reward_graded(
            None, [_ans("105")], ground_truth=["100"], answer_type=["numeric"]
        )
        assert out[0] == pytest.approx(0.25)
        assert 0.0 < out[0] < 1.0

    def test_far_numeric_is_zero(self):
        out = correctness_reward_graded(
            None, [_ans("999")], ground_truth=["100"], answer_type=["numeric"]
        )
        assert out == [0.0]

    def test_non_numeric_stays_strict(self):
        # a close-but-wrong name earns nothing; partial credit is numeric-only
        wrong = correctness_reward_graded(
            None, [_ans("E. Smith")], ground_truth=["E. Thomas"], answer_type=["name"]
        )
        right = correctness_reward_graded(
            None, [_ans("E. Thomas")], ground_truth=["E. Thomas"], answer_type=["name"]
        )
        assert wrong == [0.0]
        assert right == [1.0]


# --------------------------------------------------------------------------- #
# format_reward — shaping reward for the <think>…</think><answer>…</answer> shape
# --------------------------------------------------------------------------- #
class TestFormatReward:
    def test_full_structure_is_point_two(self):
        assert format_reward(None, ["<think>reason</think><answer>x</answer>"]) == [0.2]

    def test_answer_only_is_point_one(self):
        assert format_reward(None, ["<answer>x</answer>"]) == [0.1]

    def test_wrong_order_is_point_one(self):
        # answer present but not the rewarded think->answer structure
        assert format_reward(None, ["<answer>x</answer><think>after</think>"]) == [0.1]

    def test_no_tags_is_zero(self):
        assert format_reward(None, ["plain text, no tags at all"]) == [0.0]

    def test_accepts_chat_list(self):
        comps = [[{"role": "assistant", "content": "<think>a</think><answer>b</answer>"}]]
        assert format_reward(None, comps) == [0.2]

    def test_batch_mixed(self):
        comps = ["<think>t</think><answer>a</answer>", "<answer>only</answer>", "nothing"]
        assert format_reward(None, comps) == [0.2, 0.1, 0.0]
