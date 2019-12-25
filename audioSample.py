import wave
import pyaudio


filename = "voice/hi.wav"

wf = wave.open(filename, 'rb')

# ストリームを開く
p = pyaudio.PyAudio()
stream = p.open(
    format=p.get_format_from_width(wf.getsampwidth()),
    channels=wf.getnchannels(),
    rate=wf.getframerate(),
    output=True
)

# 音声を再生
chunk = 1024
data = wf.readframes(chunk)
while data != '':
    stream.write(data)
    data = wf.readframes(chunk)
stream.close()
p.terminate()

exit()
