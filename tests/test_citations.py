from src.citations import verify_answer


def test_verify_answer_accepts_retrieved_citation() -> None:
    result = verify_answer(
        "The standup starts at 10 AM. [schedule.md#0]",
        {"schedule.md#0"},
    )

    assert result.verified is True
    assert result.citations == ["schedule.md#0"]
    assert result.uncited_claims == []


def test_verify_answer_rejects_unknown_and_missing_citations() -> None:
    result = verify_answer(
        "The standup starts at 10 AM. [invented.md#99] The room is upstairs.",
        {"schedule.md#0"},
    )

    assert result.verified is False
    assert result.invalid_citations == ["invented.md#99"]
    assert result.uncited_claims == ["The room is upstairs."]


def test_verify_answer_allows_deterministic_no_answer() -> None:
    result = verify_answer("I do not know based on the available documents.", set())

    assert result.verified is True
    assert result.claims == []


def test_verify_answer_keeps_dotted_model_identifiers_in_one_claim() -> None:
    result = verify_answer(
        "The chat model is qwen2.5-0.5b and the embedding model is qwen3-embedding-0.6b.",
        {"tools_and_setup.md#0"},
    )

    assert len(result.claims) == 1
