import pytest

from audio_lang_tools import config


def test_new_azure_requests_stop_at_the_budget(monkeypatch):
    monkeypatch.setenv('ALTOOLS_AZURE_BUDGET', '2')
    monkeypatch.setattr(config, '_spent', config.Counter())
    config.spend('tts')
    config.spend('stt', 1.5)
    with pytest.raises(config.BudgetExceeded):
        config.spend('stt')
    assert config._spent == {'tts': 1, 'stt': 1}
