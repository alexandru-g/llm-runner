"""Inkling 2 Mini (Unsloth's GGUF repo) and what came with it in 1.16:
mainline `.gguf_file`s, DSpark drafters, the model's own sampling, and
reasoning as the model has it."""
from __future__ import annotations

import json
import types

import pytest
from click.testing import CliRunner

from llm_runner import backends, cli, launch, registry
from llm_runner import process as proc

from test_apply import host  # noqa: F401  (fixture)

# unsloth/Inkling-2-Mini-GGUF as of 2026-11-28 (abridged: two quants of ten).
INKLING_REPO = [
    ".gitattributes", "README.md", "unsloth_alternates.json", "imatrix_unsloth.gguf_file",
    "Inkling-2-Mini-UD-Q4_K_XL.gguf", "Inkling-2-Mini-UD-Q8_K_XL.gguf",
    "Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file",
    "Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file",
    "Mainline_Support/README.md", "Mainline_Support/dspark-Inkling-2-Mini-Q8_0.gguf_file",
    "Old_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file", "Old_Support/mmproj-F16.gguf_file",
    "Old_Support/dspark-Inkling-2-Mini-Q8_0.gguf_file",
    "dspark-Inkling-2-Mini-Q8_0.gguf", "dspark-Inkling-2-Mini-Q8_0_Old.gguf",
    "dspark/README.md", "dspark/dspark-Inkling-2-Mini-BF16.gguf",
    "dspark/dspark-Inkling-2-Mini-BF16-mainline.gguf_file",
    "mmproj-BF16.gguf", "mmproj-F16.gguf",
]


def test_download_offers_the_mainline_files_saved_as_gguf():
    files = cli._gguf_files(INKLING_REPO)
    assert files == {
        "Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file": "Inkling-2-Mini-UD-Q4_K_XL.gguf",
        "Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file": "Inkling-2-Mini-UD-Q8_K_XL.gguf",
        "Mainline_Support/dspark-Inkling-2-Mini-Q8_0.gguf_file":
            "dspark-Inkling-2-Mini-Q8_0.gguf",
        "dspark/dspark-Inkling-2-Mini-BF16-mainline.gguf_file":
            "dspark-Inkling-2-Mini-BF16-mainline.gguf",
        "mmproj-BF16.gguf": "mmproj-BF16.gguf",      # no mainline projector: the root ones
        "mmproj-F16.gguf": "mmproj-F16.gguf",
    }
    kinds = {n: cli._gguf_kind(n) for n in files.values()}
    assert [n for n, k in kinds.items() if k == "draft"] == [
        "dspark-Inkling-2-Mini-Q8_0.gguf", "dspark-Inkling-2-Mini-BF16-mainline.gguf"]


def test_an_ordinary_repo_is_unchanged():
    repo = ["gemma-4-26B-A4B-it-UD-Q4_K_M.gguf", "mmproj-BF16.gguf",
            "MTP/gemma-4-26B-A4B-it-MTP.gguf", "imatrix_unsloth.gguf", "README.md"]
    assert cli._gguf_files(repo) == {f: f.split("/")[-1] for f in repo[:3]}


class _Api:
    def __init__(self, names, base="thinkingmachines/Inkling-2-Mini"):
        sib = [types.SimpleNamespace(rfilename=n, size=1000) for n in names]
        self.info = types.SimpleNamespace(siblings=sib, card_data={"base_model": [base]})

    def model_info(self, repo_id, files_metadata=True):
        return self.info


def test_download_saves_renamed_files_and_the_base_models_sampling(tmp_path, monkeypatch):
    import huggingface_hub
    asked = []

    def fake_download(repo_id, filename, local_dir):
        asked.append((repo_id, filename))
        if filename == registry.GENERATION_CONFIG:
            if repo_id != "thinkingmachines/Inkling-2-Mini":
                raise FileNotFoundError(filename)
            body = '{"do_sample": true, "temperature": 1.0, "eos_token_id": 200006}'
        else:
            body = "GGUF"
        out = tmp_path / "m" / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body)
        return str(out)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: _Api(INKLING_REPO))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    r = CliRunner().invoke(cli.cli, ["download", "unsloth/Inkling-2-Mini-GGUF",
                                     "--file", "Inkling-2-Mini-UD-Q4_K_XL.gguf",
                                     "--no-mmproj", "--no-draft", "--subdir", str(tmp_path / "m")])
    assert r.exit_code == 0, r.output
    m = tmp_path / "m"
    assert sorted(p.name for p in m.iterdir() if p.is_file()) == [
        ".llmctl-source.json", "Inkling-2-Mini-UD-Q4_K_XL.gguf",
        "generation_config.json"]
    assert (m / "Inkling-2-Mini-UD-Q4_K_XL.gguf").exists()
    assert not (m / "Mainline_Support").exists()             # emptied and removed
    assert ("unsloth/Inkling-2-Mini-GGUF",
            "Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file") in asked
    assert "temp 1, top-p 1, top-k off, min-p 0" in r.output
    assert registry.sampling_args(m / "Inkling-2-Mini-UD-Q4_K_XL.gguf") == [
        "--temp", "1", "--top-p", "1", "--top-k", "0", "--min-p", "0"]


def test_dspark_drafter_is_attached_and_gets_its_spec_type(tmp_path):
    assert registry._looks_like_draft("dspark-Inkling-2-Mini-Q8_0")
    folder = tmp_path / "inkling"
    folder.mkdir()
    (folder / "Inkling-2-Mini-UD-Q4_K_XL.gguf").write_bytes(b"GGUF")
    (folder / "dspark-Inkling-2-Mini-Q8_0.gguf").write_bytes(b"GGUF")
    [spec] = registry.discover_models(tmp_path).values()
    assert spec.draft_path.name == "dspark-Inkling-2-Mini-Q8_0.gguf"
    assert registry.draft_spec_type(spec.draft_path) == "draft-dspark"
    assert registry.draft_spec_type(folder / "mtp-gemma.gguf") == "draft-mtp"


def test_sampling_keeps_what_the_model_sets(tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps(
        {"temperature": 1.0, "top_k": 64, "top_p": 0.95}))        # Gemma 4's
    gguf = tmp_path / "g.gguf"
    assert registry.sampling_args(gguf) == ["--temp", "1", "--top-p", "0.95",
                                            "--top-k", "64", "--min-p", "0"]
    assert registry.sampling_args(gguf, skip=["--temp", "0.6"])[:2] == ["--top-p", "0.95"]
    (tmp_path / "generation_config.json").write_text('{"eos_token_id": 1}')
    assert registry.sampling_args(gguf) == []                     # sets no sampling


# ------------------------------------------------------------------- run --

@pytest.fixture
def inkling(host, tmp_path, monkeypatch):  # noqa: F811
    folder = registry.MODELS_DIR / "inkling-2-mini"
    folder.mkdir()
    (folder / "Inkling-2-Mini-UD-Q4_K_XL.gguf").write_bytes(b"GGUF")
    (folder / "dspark-Inkling-2-Mini-Q8_0.gguf").write_bytes(b"GGUF")
    (folder / "mmproj-F16.gguf").write_bytes(b"GGUF")
    (folder / "generation_config.json").write_text('{"temperature": 1.0}')
    monkeypatch.setattr(backends, "find_llama_server", lambda: "/usr/bin/llama-server")
    monkeypatch.setattr(cli, "_require_backend", lambda backend: "/usr/bin/llama-server")
    return host


def _run(*args):
    r = CliRunner().invoke(cli.cli, ["run", "inkling-2-mini-ud-q4_k_xl", "--no-menu",
                                     "--port", "8120", *args])
    assert r.exit_code == 0, r.output
    rec = proc.find("inkling-2-mini-ud-q4_k_xl")
    return rec.cmd, r.output


def _after(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_run_defaults_follow_the_model(inkling):
    cmd, out = _run()
    assert "--reasoning" not in cmd and "--reasoning-budget" not in cmd   # template's own
    assert _after(cmd, "--spec-type") == "draft-dspark"
    assert [_after(cmd, f) for f in ("--temp", "--top-p", "--top-k", "--min-p")] == \
        ["1", "1", "0", "0"]
    assert "Sampling: --temp 1" in out


def test_run_reasoning_on_off_and_server_sampling(inkling):
    cmd, _ = _run("--no-reasoning", "--sampling", "server", "--spec-type", "draft-mtp")
    assert _after(cmd, "--reasoning") == "off" and "--temp" not in cmd
    assert _after(cmd, "--spec-type") == "draft-mtp"            # typed wins
    args = proc.autostart_find("inkling-2-mini-ud-q4_k_xl").run_args
    assert "--no-reasoning" in args and "--sampling" in args


def test_run_reasoning_on(inkling):
    cmd, _ = _run("--reasoning")
    assert _after(cmd, "--reasoning") == "on"


def test_restart_keeps_reasoning_off_for_launches_from_before_1_16():
    old = proc.LaunchSpec(name="g", cmd=["llama-server", "--reasoning-budget", "0"],
                          port=8081, model_path="/m/g.gguf", run_args=["g"], backend="llama")
    assert old.run_cmd() == ["run", "g", "--no-reasoning", "--port", "8081"]
    on = proc.LaunchSpec(name="g", cmd=["llama-server"], port=8081, model_path="/m/g.gguf",
                         run_args=["g", "--reasoning"], backend="llama")
    assert on.run_cmd() == ["run", "g", "--reasoning", "--port", "8081"]
    new = proc.LaunchSpec(name="g", cmd=["llama-server"], port=8081, model_path="/m/g.gguf",
                          run_args=["g"], backend="llama")
    assert new.run_cmd() == ["run", "g", "--port", "8081"]


# ------------------------------------------------------------------ menu --

def test_menu_rows_for_reasoning_sampling_and_the_drafter(inkling):
    spec = registry.discover_models(registry.MODELS_DIR)["inkling-2-mini-ud-q4_k_xl"]
    m = launch._Menu(cli.run.make_context("run", ["m"]), launch.Target(spec, "llama"))
    rows = {s.param: s for s in m.settings}
    assert [v for v, _ in rows["reasoning"].values(m.target)] == [None, True, False]
    assert m.defaults["reasoning"] is None and m.defaults["sampling"] == "model"
    assert rows["sampling"].values(m.target)[0][1] == \
        "model's (temp 1, top-p 1, top-k off, min-p 0)"
    assert rows["no_draft"].label == "Drafter"
    ctx = cli.run.make_context("run", ["m"])
    m = launch._Menu(ctx, launch.Target(spec, "llama"))
    m.commit({**m.defaults, "reasoning": False})
    assert "--no-reasoning" in cli._run_option_args(ctx)
