from audio_lang_tools.lang import cantonese as y


def test_parts_and_rhyme():
    assert y.parts('gwong2') == ('gw', 'o', 'ng')
    assert y.rhyme_at('gwong2') == 2
    assert y.rhyme_at('m4') == 0      # syllabic m: all rhyme
    assert y.rhyme_at('ngo5') == 2


def test_checked_syllables_take_tones_1_3_6():
    assert y.checked('sik6') and not y.checked('si6')
    assert y.tones_for('sap6') == (1, 3, 6)
    assert y.tones_for('si1') == (1, 2, 3, 4, 5, 6)


def test_attested_syllables_include_changed_tones():
    chars = y.chars_by_syllable()
    assert '女' in chars['neoi2']     # 仔女 zai2 neoi2
    assert '媽' in chars['maa1']
    assert 'C' not in chars.get('si1', [])


def test_gaps_are_not_sayable():
    assert not y.sayable('nin2', 'zh-HK-HiuMaanNeural')
    assert y.sayable('nin4', 'zh-HK-HiuMaanNeural')
