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


def test_heard_text_from_either_api():
    from audio_lang_tools import stt
    assert stt.text_of({'RecognitionStatus': 'Success', 'NBest': [{'Lexical': '飲 茶'}]}) == '飲茶'
    assert stt.text_of({'combinedPhrases': [{'text': '飲茶。'}]}) == '飲茶'
    assert stt.text_of({'RecognitionStatus': 'NoMatch'}) == ''
