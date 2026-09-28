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


def test_digits_are_read_every_way():
    reads = lambda t: {' '.join(r) for r in y.number_readings(t)}
    assert 'jat dim jat' in reads('1.1')            # 一點一 (個字)
    assert 'saam dim saam sap' in reads('3:30')
    assert {'sap jat', 'jat sap jat'} <= reads('11')
    assert {'ji sap jat', 'jaa jat'} <= reads('21')   # 廿一
    assert 'loeng baak ng sap' in reads('250') and 'loeng baak ng' in reads('250')   # 兩百五
    assert 'jat baak ling jat' in reads('101')
    assert 'saam maan ling ng baak' in reads('30500')
    assert {'ji', 'loeng'} == reads('2')


def test_heard_text_matches_expected_syllables():
    joined = {' '.join(r) for r in y.heard_readings('4蚊半')}
    assert 'sei man bun' in joined
    assert 'jat dim loeng go zi' in {' '.join(r) for r in y.heard_readings('一點2個字')}
