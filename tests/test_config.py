from pathlib import Path

import pytest

from jevchat.config import API_KEY_VARS, Config, ConfigError, find_config_file, load_api_key


def test_defaults_are_valid():
    Config().validate()


def test_toml_is_read_from_the_jevchat_table(tmp_path):
    (tmp_path / "jevchat.toml").write_text(
        '[jevchat]\nalphabet = "tokens"\ntemperature = 0.3\n'
    )
    config = Config.load(start_dir=tmp_path)
    assert config.alphabet == "tokens"
    assert config.temperature == 0.3
    assert config.model == "jev-latest"  # untouched default


def test_cli_overrides_beat_the_file(tmp_path):
    (tmp_path / "jevchat.toml").write_text('[jevchat]\ntemperature = 0.3\n')
    config = Config.load(start_dir=tmp_path, overrides={"temperature": 1.5, "top_p": None})
    assert config.temperature == 1.5
    assert config.top_p == Config().top_p  # None override ignored


def test_config_is_found_in_a_parent_directory(tmp_path):
    (tmp_path / "jevchat.toml").write_text('[jevchat]\nalphabet = "ascii"\n')
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert find_config_file(nested) == tmp_path / "jevchat.toml"
    assert Config.load(start_dir=nested).alphabet == "ascii"


def test_unknown_option_is_rejected(tmp_path):
    (tmp_path / "jevchat.toml").write_text('[jevchat]\nalphabt = "typo"\n')
    with pytest.raises(ConfigError, match="alphabt"):
        Config.load(start_dir=tmp_path)


@pytest.mark.parametrize(
    "kwargs",
    [{"temperature": -1}, {"top_p": 0}, {"top_p": 1.5}, {"top_k": -1},
     {"stop_bias": -0.5}, {"max_steps": 0}, {"min_steps": -1},
     {"ensemble": 0}, {"repetition_penalty": 0}, {"bisect_cutoff": 1},
     {"strategy": "bisekt"}],
)
def test_invalid_values_are_rejected(kwargs):
    with pytest.raises(ConfigError):
        Config(**kwargs).validate()


def test_strategy_error_lists_the_valid_modes():
    with pytest.raises(ConfigError, match="bisect"):
        Config(strategy="nope").validate()


def test_both_strategies_validate():
    for name in ("choice", "bisect"):
        Config(strategy=name).validate()


def test_api_key_is_read_from_an_env_file(tmp_path, monkeypatch):
    for var in API_KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    env = tmp_path / ".env"
    env.write_text('api_key="jev_secret_value"\n')
    assert load_api_key(env_file=env) == "jev_secret_value"


def test_env_file_wins_over_an_exported_value(tmp_path, monkeypatch):
    monkeypatch.setenv("api_key", "stale")
    env = tmp_path / ".env"
    env.write_text('api_key="fresh"\n')
    assert load_api_key(env_file=env) == "fresh"


def test_missing_key_explains_where_to_put_one(tmp_path, monkeypatch):
    for var in API_KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    empty = tmp_path / ".env"
    empty.write_text("\n")
    with pytest.raises(ConfigError, match=r"\.env"):
        load_api_key(env_file=empty)


# --- resolution: settings that depend on the alphabet and the strategy ---------

def _alpha(size, **defaults):
    from jevchat.alphabet import Alphabet, Symbol
    return Alphabet(name="x", description="",
                    symbols=tuple(Symbol(f"k{i}", "x") for i in range(size - 1)),
                    defaults=tuple(sorted(defaults.items())))


def test_an_alphabet_can_declare_preferences():
    resolved = Config().resolve(_alpha(50, repetition_penalty=1.0))
    assert resolved.repetition_penalty == 1.0


def test_an_explicit_setting_beats_the_alphabet():
    cfg = Config.load(overrides={"repetition_penalty": 1.4})
    assert cfg.resolve(_alpha(50, repetition_penalty=1.0)).repetition_penalty == 1.4


def test_the_config_file_also_counts_as_explicit(tmp_path):
    (tmp_path / "jevchat.toml").write_text("[jevchat]\nrepetition_penalty = 1.4\n")
    cfg = Config.load(start_dir=tmp_path)
    assert cfg.resolve(_alpha(50, repetition_penalty=1.0)).repetition_penalty == 1.4


def test_refine_raises_bucket_size_so_the_probe_fits_one_question():
    # 49,861 symbols cannot be probed as 393 buckets of 127.
    resolved = Config(strategy="refine").resolve(_alpha(49862))
    assert resolved.bucket_size == 197
    assert -(-49861 // resolved.bucket_size) <= 254


def test_refine_leaves_bucket_size_alone_when_it_already_fits():
    assert Config(strategy="refine").resolve(_alpha(1122)).bucket_size == 127


def test_an_explicit_bucket_size_is_not_raised():
    cfg = Config.load(overrides={"strategy": "refine", "bucket_size": 127})
    assert cfg.resolve(_alpha(49862)).bucket_size == 127


def test_buckets_does_not_need_the_probe_to_fit():
    assert Config(strategy="buckets").resolve(_alpha(49862)).bucket_size == 127


def test_with_overrides_marks_them_explicit():
    cfg = Config().with_overrides(repetition_penalty=1.4)
    assert "repetition_penalty" in cfg.explicit
    assert cfg.resolve(_alpha(50, repetition_penalty=1.0)).repetition_penalty == 1.4


def test_resolution_still_validates():
    with pytest.raises(ConfigError):
        Config().resolve(_alpha(50, top_p=9.0))
