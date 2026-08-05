import pytest

from atlasfin.contracts import make_chunk_id, parse_chunk_id, slugify_section


def test_round_trip():
    cid = make_chunk_id("3M_2018_10K", slugify_section("Item 7 > Liquidity"), "some chunk text")
    doc_name, section_slug, text_hash = parse_chunk_id(cid)
    assert doc_name == "3M_2018_10K"
    assert section_slug == "item-7-liquidity"
    assert len(text_hash) == 12


def test_same_text_same_hash():
    cid1 = make_chunk_id("d", "s", "identical text")
    cid2 = make_chunk_id("d", "s", "identical text")
    assert cid1 == cid2


def test_different_text_different_hash():
    cid1 = make_chunk_id("d", "s", "text one")
    cid2 = make_chunk_id("d", "s", "text two")
    assert cid1 != cid2


def test_slugify_empty_falls_back_to_root():
    assert slugify_section("") == "root"
    assert slugify_section("   ") == "root"


def test_slugify_strips_punctuation():
    assert slugify_section("Item 7A: Risk Factors!!!") == "item-7a-risk-factors"


def test_parse_malformed_raises():
    with pytest.raises(ValueError):
        parse_chunk_id("not-a-valid-chunk-id")
