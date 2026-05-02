from src.tv1_data.parse_clean import parse_source_note


def test_parse_source_note_infers_issuer_from_thong_tu_code() -> None:
    metadata = parse_source_note(
        "(\u0110i\u1ec1u 2 Th\u00f4ng t\u01b0 s\u1ed1 27/2010/TT-BGTVT, c\u00f3 hi\u1ec7u l\u1ef1c thi h\u00e0nh k\u1ec3 t\u1eeb ng\u00e0y 15/09/2010)"
    )

    assert metadata.law_id == "Th\u00f4ng t\u01b0 s\u1ed1 27/2010/TT-BGTVT"
    assert metadata.issuer == "B\u1ed9 Giao th\u00f4ng v\u1eadn t\u1ea3i"
    assert metadata.effective_date == "15/09/2010"


def test_parse_source_note_infers_issuer_from_quoc_hoi_code() -> None:
    metadata = parse_source_note(
        "(\u0110i\u1ec1u 1 Lu\u1eadt s\u1ed1 57/2020/QH14, c\u00f3 hi\u1ec7u l\u1ef1c thi h\u00e0nh k\u1ec3 t\u1eeb ng\u00e0y 01/01/2021)"
    )

    assert metadata.law_id == "Lu\u1eadt s\u1ed1 57/2020/QH14"
    assert metadata.issuer == "Qu\u1ed1c h\u1ed9i"
    assert metadata.effective_date == "01/01/2021"


def test_parse_source_note_infers_multiple_issuers_for_joint_circular() -> None:
    metadata = parse_source_note(
        "(\u0110i\u1ec1u 1 Th\u00f4ng t\u01b0 li\u00ean t\u1ecbch s\u1ed1 01/2016/TTLT-TANDTC-VKSNDTC-BTP, c\u00f3 hi\u1ec7u l\u1ef1c thi h\u00e0nh k\u1ec3 t\u1eeb ng\u00e0y 01/02/2016)"
    )

    assert metadata.issuer == (
        "T\u00f2a \u00e1n nh\u00e2n d\u00e2n t\u1ed1i cao; "
        "Vi\u1ec7n ki\u1ec3m s\u00e1t nh\u00e2n d\u00e2n t\u1ed1i cao; "
        "B\u1ed9 T\u01b0 ph\u00e1p"
    )


def test_parse_source_note_prefers_explicit_issuer_in_note() -> None:
    metadata = parse_source_note(
        "(\u0110i\u1ec1u 1 Lu\u1eadt s\u1ed1 57/2020/QH14 ng\u00e0y 16/6/2020 c\u1ee7a Qu\u1ed1c h\u1ed9i, c\u00f3 hi\u1ec7u l\u1ef1c thi h\u00e0nh k\u1ec3 t\u1eeb ng\u00e0y 01/01/2021)"
    )

    assert metadata.issuer == "Qu\u1ed1c h\u1ed9i"


def test_parse_source_note_infers_issuer_from_ttg_title_fragment() -> None:
    metadata = parse_source_note(
        "(\u0110i\u1ec1u 2 Quy\u1ebft \u0111\u1ecbnh s\u1ed1 01/2012/Q\u0110 -TTg, c\u00f3 hi\u1ec7u l\u1ef1c thi h\u00e0nh k\u1ec3 t\u1eeb ng\u00e0y 25/02/2012)"
    )

    assert metadata.issuer == "Th\u1ee7 t\u01b0\u1edbng Ch\u00ednh ph\u1ee7"


def test_parse_source_note_infers_issuer_from_note_when_law_id_missing() -> None:
    metadata = parse_source_note(
        "(\u0110i\u1ec1u 10 Ph\u00e1p l\u1ec7nh V\u1ec1 H\u00e0m, c\u1ea5p ngo\u1ea1i giao, c\u00f3 hi\u1ec7u l\u1ef1c thi h\u00e0nh k\u1ec3 t\u1eeb ng\u00e0y 31/05/1995)"
    )

    assert metadata.issuer == "\u1ee6y ban Th\u01b0\u1eddng v\u1ee5 Qu\u1ed1c h\u1ed9i"


def test_parse_source_note_parses_textual_effective_date() -> None:
    metadata = parse_source_note(
        "(Th\u00f4ng t\u01b0 n\u00e0y c\u00f3 hi\u1ec7u l\u1ef1c thi h\u00e0nh k\u1ec3 t\u1eeb ng\u00e0y 28 th\u00e1ng 6 n\u0103m 2017. B\u00e3i b\u1ecf c\u00e1c v\u0103n b\u1ea3n sau...)"
    )

    assert metadata.effective_date == "28/06/2017"


def test_parse_source_note_infers_hdtp_issuer() -> None:
    metadata = parse_source_note(
        "(\u0110i\u1ec1u 1 Ngh\u1ecb quy\u1ebft s\u1ed1 01/2016/NQ.H\u0110TP, c\u00f3 hi\u1ec7u l\u1ef1c thi h\u00e0nh k\u1ec3 t\u1eeb ng\u00e0y 01/07/2016)"
    )

    assert metadata.issuer == "H\u1ed9i \u0111\u1ed3ng Th\u1ea9m ph\u00e1n T\u00f2a \u00e1n nh\u00e2n d\u00e2n t\u1ed1i cao"
