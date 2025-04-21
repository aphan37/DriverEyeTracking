# voice_alert.py
import pyttsx3

def speak_alert(text="Please wake up! You seem drowsy!"):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()
