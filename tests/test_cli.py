from pathlib import Path

import pytest

from src import cli


def test_parse_upload_command_supports_quoted_multiple_windows_paths() -> None:
    command = "/upload 'C:\\folder with spaces\\one.pdf' \"D:\\other folder\\two.docx\""

    assert cli.parse_upload_command(command) == [
        Path(r"C:\folder with spaces\one.pdf"),
        Path(r"D:\other folder\two.docx"),
    ]
    assert cli.parse_upload_command("What is in the document?") is None
    assert cli.parse_upload_command("/uploader one.pdf") is None


def test_parse_clean_command_accepts_optional_paths() -> None:
    assert cli.parse_clean_command("/clean") == []
    assert cli.parse_clean_command("/clean 'C:\\fresh docs'") == [
        Path(r"C:\fresh docs")
    ]
    assert cli.parse_clean_command("/cleanup") is None


def test_cli_upload_command_does_not_invoke_chat_model(monkeypatch: pytest.MonkeyPatch) -> None:
    uploads: list[list[Path]] = []

    def fake_upload(paths: list[Path]) -> dict[str, object]:
        uploads.append(paths)
        return {
            "staged": [
                {"document_name": "one.pdf"},
                {"document_name": "two.pdf"},
            ],
            "documents": [],
            "indexed_files": 2,
            "skipped_files": 0,
            "error_files": 0,
        }

    monkeypatch.setattr(cli, "add_document_files", fake_upload)
    monkeypatch.setattr(
        cli,
        "answer_query",
        lambda _question: pytest.fail("upload commands must not invoke the chat model"),
    )
    inputs = iter(["/upload 'C:\\one.pdf' 'C:\\two.pdf'", "exit"])

    cli.run_cli(lambda _prompt: next(inputs))

    assert uploads == [[Path(r"C:\one.pdf"), Path(r"C:\two.pdf")]]


def test_cli_clean_command_clears_without_invoking_chat_model(monkeypatch: pytest.MonkeyPatch) -> None:
    clean_calls: list[bool] = []

    def fake_reset() -> dict[str, object]:
        clean_calls.append(True)
        return {"removed_uploads": 2, "final_rows": 0, "final_vectors": 0}

    monkeypatch.setattr(cli, "reset_knowledge_base", fake_reset)
    monkeypatch.setattr(
        cli,
        "answer_query",
        lambda _question: pytest.fail("clean commands must not invoke the chat model"),
    )
    inputs = iter(["/clean", "exit"])

    cli.run_cli(lambda _prompt: next(inputs))

    assert clean_calls == [True]


def test_cli_clean_with_paths_replaces_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    replacements: list[list[Path]] = []

    def fake_replace(paths: list[Path]) -> dict[str, object]:
        replacements.append(paths)
        return {
            "reset": {"removed_uploads": 1, "final_rows": 0, "final_vectors": 0},
            "staged": [{"document_name": "fresh.md"}],
            "documents": [{"document_name": "fresh.md", "status": "indexed", "chunks": 1, "vectors": 1}],
            "indexed_files": 1,
            "skipped_files": 0,
            "error_files": 0,
        }

    monkeypatch.setattr(cli, "replace_document_files", fake_replace)
    monkeypatch.setattr(
        cli,
        "answer_query",
        lambda _question: pytest.fail("clean commands must not invoke the chat model"),
    )
    inputs = iter(["/clean 'C:\\fresh\\docs'", "exit"])

    cli.run_cli(lambda _prompt: next(inputs))

    assert replacements == [[Path(r"C:\fresh\docs")]]
