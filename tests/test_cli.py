import io
import subprocess
import sys

import pytest
from helpers import LIGHTHOUSE_STORY, FakeImageGenerator, FakeLLM, storyboard_json

from storyboard_weaver import cli
from storyboard_weaver.config import ENV_VARS
from storyboard_weaver.errors import ProviderError


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Run every CLI test in an empty folder with none of the app's variables set."""
    for var in ENV_VARS.values():
        monkeypatch.setenv(var, "")  # records the original value so it is restored
        monkeypatch.delenv(var)  # ...including values a .env file sets during the test
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))


@pytest.fixture
def fake_llm(monkeypatch):
    llm = FakeLLM([storyboard_json(3)])
    seen = {}

    def build(settings):
        seen["settings"] = settings
        settings.api_key(settings.llm_provider)  # keep the real "missing key" behaviour
        return llm

    monkeypatch.setattr(cli, "build_llm", build)
    llm.seen = seen
    return llm


def test_generate_end_to_end(monkeypatch, tmp_path, fake_llm, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    code = cli.main(["generate", LIGHTHOUSE_STORY, "--scenes", "3", "--style", "noir"])

    assert code == 0
    out = capsys.readouterr().out
    assert "The Keeper's Letter (mystery)" in out
    assert "two-shot" in out
    runs = [p for p in (tmp_path / "outputs").iterdir() if p.is_dir()]
    assert len(runs) == 1
    assert {p.name for p in runs[0].iterdir()} == {
        "storyboard.json",
        "storyboard.md",
        "mood_chart.png",
    }
    assert (tmp_path / "outputs" / "knowledge_base.json").exists()


def test_missing_key_is_a_configuration_error(fake_llm, capsys):
    assert cli.main(["generate", "A story"]) == 2
    assert "GEMINI_API_KEY is not set" in capsys.readouterr().err
    assert fake_llm.prompts == []


def test_dotenv_file_in_the_working_directory_is_loaded(tmp_path, fake_llm):
    (tmp_path / ".env").write_text(
        "GEMINI_API_KEY=from-dotenv\nSTORYBOARD_OUTPUT_DIR=runs\n", encoding="utf-8"
    )
    assert cli.main(["generate", LIGHTHOUSE_STORY, "-n", "3"]) == 0
    assert fake_llm.seen["settings"].gemini_api_key == "from-dotenv"
    assert (tmp_path / "runs").is_dir()


def test_command_line_overrides(monkeypatch, tmp_path, fake_llm):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    monkeypatch.setenv("STORYBOARD_LLM_MODEL", "gemini-model-from-env")
    images = FakeImageGenerator()
    monkeypatch.setattr(cli, "build_image_generator", lambda settings: images)

    code = cli.main(
        ["generate", LIGHTHOUSE_STORY, "-n", "3", "--provider", "openai", "--images", "openai",
         "-o", str(tmp_path / "elsewhere"), "--no-memory"]
    )  # fmt: skip

    assert code == 0
    settings = fake_llm.seen["settings"]
    assert settings.llm_provider == "openai"
    assert settings.text_model != "gemini-model-from-env"  # dropped with the provider switch
    assert len(images.prompts) == 3
    assert not (tmp_path / "elsewhere" / "knowledge_base.json").exists()


def test_story_from_file_and_from_stdin(monkeypatch, tmp_path, fake_llm):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    story_file = tmp_path / "story.txt"
    story_file.write_text("A story from a file about a lighthouse.", encoding="utf-8")
    fake_llm.replies.append(storyboard_json(3))

    assert cli.main(["generate", "--file", str(story_file), "-n", "3"]) == 0
    assert "A story from a file" in fake_llm.prompts[0]

    monkeypatch.setattr(sys, "stdin", io.StringIO("A story piped in on stdin."))
    assert cli.main(["generate", "-", "-n", "3"]) == 0
    assert "A story piped in on stdin." in fake_llm.prompts[1]


def test_bad_input_exits_with_usage_code(monkeypatch, tmp_path, fake_llm, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert cli.main(["generate", "--file", str(tmp_path / "missing.txt")]) == 2
    assert "Could not read" in capsys.readouterr().err
    assert cli.main(["generate", "A story", "--scenes", "40"]) == 2
    assert "between 1 and 12" in capsys.readouterr().err
    assert cli.main(["generate", "   "]) == 2
    assert "The story is empty" in capsys.readouterr().err


def test_model_failure_exits_with_error_code(monkeypatch, fake_llm, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    fake_llm.replies[:] = [ProviderError("gemini: HTTP 503: overloaded")]
    assert cli.main(["generate", "A story"]) == 1
    err = capsys.readouterr().err
    assert "Error: The language model request failed. gemini: HTTP 503: overloaded" in err
    assert "Traceback" not in err


def test_styles_command(capsys):
    assert cli.main(["styles"]) == 0
    out = capsys.readouterr().out
    for name in ("cinematic", "documentary", "anime", "noir", "experimental"):
        assert name in out


def test_history_command(monkeypatch, fake_llm, capsys):
    assert cli.main(["history"]) == 0
    assert "No storyboards stored yet" in capsys.readouterr().out

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    cli.main(["generate", LIGHTHOUSE_STORY, "-n", "3"])
    capsys.readouterr()
    assert cli.main(["history"]) == 0
    assert "The Keeper's Letter  [mystery, cinematic]" in capsys.readouterr().out


def test_python_dash_m_entry_point():
    completed = subprocess.run(
        [sys.executable, "-m", "storyboard_weaver", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.startswith("storyboard-weaver ")
