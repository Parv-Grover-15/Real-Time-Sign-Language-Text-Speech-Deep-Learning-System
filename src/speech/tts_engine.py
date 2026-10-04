"""
Non-blocking local Text-To-Speech Engine (from Sign-Bridge contribution)
Uses pyttsx3 in a background thread to prevent UI freezing.
"""
import threading
import pyttsx3
from config.settings import TTS_RATE, TTS_VOLUME

class TTSEngine:
    def __init__(self, rate=TTS_RATE, volume=TTS_VOLUME):
        self.rate = rate
        self.volume = volume
        self.lock = threading.Lock()
        self._init_engine()

    def _init_engine(self):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)
            self.available = True
        except Exception as e:
            print(f"[TTSEngine] pyttsx3 initialization warning: {e}")
            self.engine = None
            self.available = False

    def speak(self, text: str):
        """
        Speaks text in a non-blocking background thread.
        """
        if not text or not text.strip():
            return

        text = text.strip()

        def _run():
            with self.lock:
                try:
                    if self.engine is None:
                        self._init_engine()
                    if self.engine:
                        self.engine.say(text)
                        self.engine.runAndWait()
                except Exception as e:
                    print(f"[TTSEngine] Speech synthesis error: {e}")
                    # Re-initialize engine for next call if COM failed
                    self._init_engine()

        threading.Thread(target=_run, daemon=True).start()
