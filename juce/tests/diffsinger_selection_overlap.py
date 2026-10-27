"""Selection-scoped overlap checks and real DS synthesis (source, packages, bank, output)."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8')
source, packages, voice, output = map(lambda p: Path(p).resolve(), sys.argv[1:5])
output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(packages))
spec = importlib.util.spec_from_file_location('bridge', source)
b = importlib.util.module_from_spec(spec)
sys.modules['bridge'] = b
spec.loader.exec_module(b)
import soundfile as sf
checks = []

def check(name, passed):
    assert passed, name
    checks.append(name)
    print('PASS', name, flush=True)

def note(key, start, duration, lyric='ni', audible=False):
    return dict(id=key, start=start, duration=duration, lyric=lyric, midi=60, audible=audible)

notes = [note('a', .5, .4, audible=True), note('b', .9, .4, 'hao', True),
         note('overlap1', 2, 1), note('overlap2', 2.2, .3),
         note('after', 3.5, .5, 'ma')]
ids = lambda phrases: [[n['id'] for n in p] for p in phrases]
check('remote_overlap_does_not_enter_selected_phrase', ids(b.groups(notes, {'a'})) == [['a', 'b']])
check('context_kept_on_both_sides_of_selected_note', ids(b.groups(notes, {'b'})) == [['a', 'b']])
check('selection_after_overlap_is_independent', ids(b.groups(notes, {'after'})) == [['after']])
check('multiple_safe_selections_separated_by_overlap', ids(b.groups(notes, {'a', 'after'})) == [['a', 'b'], ['after']])
check('empty_scope_does_not_warn', b.groups(notes, set()) == [])
for selected in (None, {'overlap1'}, {'overlap2'}, {'a', 'overlap2'}):
    try:
        b.groups(notes, selected)
    except ValueError as e:
        check('selected_or_whole_overlap_still_rejected_' + str(selected), '重叠' in str(e))
    else:
        raise AssertionError('selected overlap was accepted')
nested = [note('long', 1, 2), note('short', 1.2, .1), note('inside', 2, .1), note('safe', 3, .4)]
check('nested_overlap_uses_maximum_end', ids(b.groups(nested, {'safe'})) == [['safe']])
try:
    b.groups(nested, {'inside'})
except ValueError:
    check('nested_selected_overlap_warns', True)
else:
    raise AssertionError('nested overlap accepted')

bank = b.Bank(voice)
def request(values, operation='render', **extra):
    return dict(operation=operation, voicebank=str(voice), language='zh', duration=4.4,
                notes=copy.deepcopy(values), **extra)
baseline = b.execute(request(notes[:2]), output/'baseline.json', bank)
selected = b.execute(request(notes), output/'selected.json', bank)
reference, sr = sf.read(output/'baseline.wav')
actual, actual_sr = sf.read(output/'selected.wav')
check('real_DS_selected_phrase_synthesizes', selected['ok'] and b.np.isfinite(actual).all()
      and b.np.max(b.np.abs(actual)) > .001)
check('unselected_overlap_does_not_change_selected_audio', sr == actual_sr
      and reference.shape == actual.shape and b.np.max(b.np.abs(reference-actual)) < 1e-6)
check('unselected_later_notes_remain_silent', b.np.max(b.np.abs(actual[int(2*sr):])) == 0)
retake = b.execute(request(notes, 'pitch', retake_ids=['a']), output/'pitch.json', bank)
check('selected_pitch_generation_ignores_unrelated_overlap', retake['ok'] and [c['id'] for c in retake['curves']] == ['a'])
baseline_timing = b.execute(request(notes[:2], 'timing'), output/'baseline-timing.json', bank)
for scope in (['a'], ['b'], ['after'], ['a', 'after']):
    result = b.execute(request(notes, 'timing', retake_ids=scope), output/'timing.json', bank)
    expected = {'a', 'b'} if scope != ['after'] else set()
    if 'after' in scope: expected.add('after')
    actual_ids = {r['id'] for r in result['phonemes'] if r['id']}
    check('timing_scope_and_context_' + str(scope), result['ok'] and actual_ids == expected)
    if scope in (['a'], ['b']):
        check('timing_unchanged_by_remote_overlap_' + str(scope), result['phonemes'] == baseline_timing['phonemes'])
for scope in (None, ['overlap1'], ['overlap2'], ['a', 'overlap2']):
    extra = {} if scope is None else {'retake_ids': scope}
    try:
        b.execute(request(notes, 'timing', **extra), output/'bad-timing.json', bank)
    except ValueError as e:
        check('timing_actual_overlap_still_warns_' + str(scope), '重叠' in str(e))
    else:
        raise AssertionError('overlapping timing request accepted')
bad = copy.deepcopy(notes)
bad[2]['audible'] = True
try:
    b.execute(request(bad), output/'bad.json', bank)
except ValueError as e:
    check('actual_render_warns_when_overlap_selected', '重叠' in str(e))
else:
    raise AssertionError('actual overlapping render accepted')
remote_bad = note('unrelated_bad_reading', 6, .4, '[missing_phoneme_229]')
for operation in ('timing', 'pitch', 'parameters', 'phonemize', 'render'):
    req = request(notes + [remote_bad], operation, retake_ids=['a'])
    req['duration'] = 7
    result = b.execute(req, output/('scope-' + operation + '.json'), bank)
    check('remote_bad_reading_does_not_block_' + operation, result['ok'])
    if operation in ('pitch', 'parameters', 'phonemize'):
        key = dict(pitch='curves', parameters='parameters', phonemize='pronunciations')[operation]
        check('only_selected_results_' + operation, {r['id'] for r in result[key]} == {'a'})
    req['retake_ids'] = [remote_bad['id']]
    if operation == 'render':
        for n in req['notes']: n['audible'] = n['id'] == remote_bad['id']
    try:
        b.execute(req, output/'selected-bad-reading.json', bank)
    except ValueError as e:
        check('selected_bad_reading_still_rejected_' + operation, 'missing_phoneme_229' in str(e))
    else:
        raise AssertionError('selected invalid reading accepted')
# Pronunciation editing is allowed even when note timing overlaps.
result = b.execute(request(notes, 'phonemize', retake_ids=['overlap1']), output/'overlap-reading.json', bank)
check('overlapping_note_can_edit_reading', [r['id'] for r in result['pronunciations']] == ['overlap1'])
music = [note('music1', .5, .4, '音'), note('music2', .9, .4, '乐')]
full = b.execute(request(music, 'phonemize'), output/'word-full.json', bank)
part = b.execute(request(music + [remote_bad], 'phonemize', retake_ids=['music2']), output/'word-part.json', bank)
check('Chinese_word_reading_context_preserved', part['pronunciations'] == [full['pronunciations'][1]])
long_word = [note(str(i), i*.5, .5, lyric) for i, lyric in enumerate(['啊']*30+['音', '乐'])]
full = b.execute(request(long_word, 'phonemize'), output/'long-word-full.json', bank)
part = b.execute(request(long_word, 'phonemize', retake_ids=['31']), output/'long-word-part.json', bank)
check('Chinese_word_context_across_inference_boundary', part['pronunciations'] == [full['pronunciations'][-1]])
continuation = [note('word', .5, .4, '[zh/n zh/i zh/h zh/ao]'), note('next', .9, .4, '+2')]
full = b.execute(request(continuation, 'timing'), output/'continuation-full.json', bank)
part = b.execute(request(continuation + [remote_bad], 'timing', retake_ids=['next']), output/'continuation-part.json', bank)
check('multi_syllable_continuation_context_preserved', part['phonemes'] == full['phonemes'])
for operation in ('phonemize', 'timing', 'pitch', 'parameters'):
    for chosen in ([], ['missing']):
        try:
            b.execute(request(notes, operation, retake_ids=chosen), output/'invalid-scope.json', bank)
        except ValueError as e:
            check('invalid_scope_never_becomes_whole_track_' + operation + str(chosen), '所选音符无效' in str(e))
        else:
            raise AssertionError('invalid scope accepted')
(output/'report.json').write_text(json.dumps(dict(passed=True, checks=checks, sample_rate=sr), ensure_ascii=False, indent=2), encoding='utf-8')
