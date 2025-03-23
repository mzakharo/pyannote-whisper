from pyannote_whisper.utils import diarize_text

file = "data/clip1.wav"


location = '.cache'
from joblib import Memory
memory = Memory(location, verbose=0)


@memory.cache
def get_asr(data, model):
    from faster_whisper import WhisperModel
    model = WhisperModel(model, device="cpu", compute_type="int8")

    segs, _ = model.transcribe(data, beam_size=5, word_timestamps=True)

    l = []
    for seg in segs:
        print(f'{seg.start:.2f} --> {seg.end:.2f} {seg.text}')
        l.append(seg)
    return dict(segments=l)
    import whisper
    model = whisper.load_model(model)
    return model.transcribe(data)

@memory.cache
def get_diarization(data):
    from pyannote.audio import Pipeline
    import my_token
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token=my_token.token)
    return pipeline(data)
    

print('get_asr')
asr_result = get_asr(file, "Systran/faster-whisper-medium.en")
print('get_diarization')
diarization_result = get_diarization(file)
final_result = diarize_text(asr_result, diarization_result)

for seg, spk, sent in final_result:
    line = f'{seg.start:.2f} --> {seg.end:.2f} {spk} {sent}'
    print(line)