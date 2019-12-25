import subprocess


class Player:
    def __init__(self):
        self.voiceFile = ""
        self.isPlaying = False

    def set(self, voiceFile):
        self.voiceFile = voiceFile

    def play(self):
        if self.voiceFile == "":
            return
        print('play ' + self.voiceFile)
        subprocess.call(["aplay", self.voiceFile])  # 再生
