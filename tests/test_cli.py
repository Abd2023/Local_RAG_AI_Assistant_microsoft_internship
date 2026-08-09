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
