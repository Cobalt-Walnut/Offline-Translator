from gpiozero import LED, Button, DigitalInputDevice
import time
import queue
import numpy as np
import sounddevice as sd
import sys
import json
import vosk
import ctranslate2
import sentencepiece as spm
import subprocess
import os
import gc
from deepmultilingualpunctuation import PunctuationModel

# --- Audio Device Settings (with automatic runtime detection) ---

# Output audio devices (speaker)
POSSIBLE_DEVICES = ["plughw:3,0", "plughw:2,0"]
APLAY_DEVICE = None  # to be selected below

def test_audio_device(device):
    """
    Try playing a short silence to test if device exists and is usable.
    Returns True if successful, False otherwise.
    """
    try:
        silence = b'\x00\x00' * (16000 // 10)  # 0.1 seconds silence at 16kHz, mono, 16-bit
        proc = subprocess.run(
            ['aplay', '-D', device, '-f', 'S16_LE', '-r', '16000'],
            input=silence,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2
        )
        return proc.returncode == 0
    except Exception:
        return False

def select_audio_device():
    global APLAY_DEVICE
    for device in POSSIBLE_DEVICES:
        if test_audio_device(device):
            APLAY_DEVICE = device
            print(f"Selected output audio device: {APLAY_DEVICE}")
            return
    APLAY_DEVICE = "default"
    print("No valid output device found, falling back to 'default'")

select_audio_device()

# Input audio devices (microphone)
POSSIBLE_INPUT_DEVICES = ["plughw:3,0", "plughw:2,0"]
INPUT_DEVICE = None  # to be selected below

def test_input_device(device):
    """
    Try to open a small input stream to test if the device is valid for recording.
    Returns True if successful, False otherwise.
    """
    try:
        with sd.InputStream(device=device, channels=1, samplerate=16000):
            pass
        return True
    except Exception:
        return False

def select_input_device():
    global INPUT_DEVICE
    for device in POSSIBLE_INPUT_DEVICES:
        if test_input_device(device):
            INPUT_DEVICE = device
            print(f"Selected input audio device: {INPUT_DEVICE}")
            return
    INPUT_DEVICE = None  # fallback to default if none found
    print("No valid input device found; will use default input device")

select_input_device()

# --- GPIO Setup ---
RED_LED_PIN = 22
GREEN_LED_PIN = 27
BUTTON_PIN = 4
SWITCH_PIN = 13
EXIT_BUTTON_PIN = 5  # Exit button on GPIO5

red_led = LED(RED_LED_PIN)
green_led = LED(GREEN_LED_PIN)
record_button = Button(BUTTON_PIN, pull_up=True)
direction_switch = DigitalInputDevice(SWITCH_PIN, pull_up=False)
exit_button = Button(EXIT_BUTTON_PIN, pull_up=True, bounce_time=0.1)  # Debounced

# --- Model Paths ---
VOSK_EN = "vosk_models/vosk-model-small-en-us-0.15"
VOSK_ES = "vosk_models/vosk-model-small-es-0.42"
CT_EN_ES = "./CTranslate/en-es-ctranslate2"
CT_ES_EN = "./CTranslate/es-en-ctranslate2"
SP_EN_ES_SRC = "./CTranslate/opus-mt-en-es/source.spm"
SP_EN_ES_TGT = "./CTranslate/opus-mt-en-es/target.spm"
SP_ES_EN_SRC = "./CTranslate/opus-mt-es-en/source.spm"
SP_ES_EN_TGT = "./CTranslate/opus-mt-es-en/target.spm"
PIPER_ES = "./piper_models/es_ES-davefx-medium.onnx"
PIPER_EN = "./piper_models/en_US-hfc_male-medium.onnx"
PIPER_CONFIG_ES = "./piper_models/es_ES-davefx-medium.onnx.json"
PIPER_CONFIG_EN = "./piper_models/en_US-hfc_male-medium.onnx.json"
PIPER_BIN = "./piper_models/piper/piper"

# Pre-recorded announcement files
MODE_EN_ES_FILE = "./en-es-mode-quieter.wav"
MODE_ES_EN_FILE = "./es-en-mode-quieter.wav"
NO_AUDIO_EN_FILE = "./no_audio-en-quieter.wav"
NO_AUDIO_ES_FILE = "./no_audio-es-quieter.wav"
EXIT_FILE = "./exit1.wav"

# --- Global Model Holders ---
current_direction = None
vosk_model = None
translator = None
sp_source = None
sp_target = None
piper_voice = None
piper_config = None

# --- Punctuation Model ---
punct_model = PunctuationModel(model="kredor/punctuate-all")

# --- Exit flag ---
exit_requested = False

#time.sleep(7)  # Allow some time for audio devices to initialize at startup

def exit_program():
    global exit_requested
    for _ in range(8):
        red_led.toggle()
        time.sleep(0.25)
    play_audio_file(EXIT_FILE)
    exit_requested = True
    print("\nExit button pressed. Cleaning up...")

exit_button.when_pressed = exit_program

def load_models(direction):
    global vosk_model, translator, sp_source, sp_target, piper_voice, piper_config
    unload_models()
    if direction == "en_to_es":
        print("Loading EN->ES models")
        vosk_model = vosk.Model(VOSK_EN)
        translator = ctranslate2.Translator(CT_EN_ES, device="cpu")
        sp_source = spm.SentencePieceProcessor()
        sp_source.load(SP_EN_ES_SRC)
        sp_target = spm.SentencePieceProcessor()
        sp_target.load(SP_EN_ES_TGT)
        piper_voice = PIPER_ES
        piper_config = PIPER_CONFIG_ES
    elif direction == "es_to_en":
        print("Loading ES->EN models")
        vosk_model = vosk.Model(VOSK_ES)
        translator = ctranslate2.Translator(CT_ES_EN, device="cpu")
        sp_source = spm.SentencePieceProcessor()
        sp_source.load(SP_ES_EN_SRC)
        sp_target = spm.SentencePieceProcessor()
        sp_target.load(SP_ES_EN_TGT)
        piper_voice = PIPER_EN
        piper_config = PIPER_CONFIG_EN

def unload_models():
    global vosk_model, translator, sp_source, sp_target
    vosk_model = None
    translator = None
    sp_source = None
    sp_target = None
    gc.collect()

def get_translation_direction():
    return "es_to_en" if direction_switch.value == 1 else "en_to_es"

def play_audio_file(file_path):
    if os.path.exists(file_path):
        subprocess.run(["aplay", "-D", APLAY_DEVICE, file_path])
    else:
        print(f"Audio file missing: {file_path}")

def play_mode_announcement(direction):
    file_path = MODE_ES_EN_FILE if direction == "es_to_en" else MODE_EN_ES_FILE
    play_audio_file(file_path)

def play_no_audio(direction):
    if direction == "en_to_es":
        play_audio_file(NO_AUDIO_EN_FILE)
    else:
        play_audio_file(NO_AUDIO_ES_FILE)

def transcribe(audio, samplerate):
    rec = vosk.KaldiRecognizer(vosk_model, samplerate)
    rec.AcceptWaveform(audio.tobytes())
    result = json.loads(rec.FinalResult())
    return result.get("text", "")

def translate(text):
    tokens = sp_source.encode(text, out_type=str)
    tokens.append("</s>")
    results = translator.translate_batch([tokens])
    return sp_target.decode(results[0].hypotheses[0])

def speak(text):
    piper_proc = subprocess.Popen(
        [PIPER_BIN, "--model", piper_voice, "--config", piper_config, "--output-raw"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE
    )
    aplay_proc = subprocess.Popen(
        ["aplay", "-D", APLAY_DEVICE, "-r", "22050", "-f", "S16_LE", "-t", "raw"],
        stdin=piper_proc.stdout
    )
    piper_proc.stdin.write(text.encode("utf-8"))
    piper_proc.stdin.close()
    aplay_proc.wait()
    piper_proc.wait()

def wait_for_button_or_switch_change(last_direction):
    while not exit_requested:
        if record_button.is_active:
            return None
        new_direction = get_translation_direction()
        if new_direction != last_direction:
            return new_direction
        time.sleep(0.01)
    return "exit"

def wait_for_button_release_or_switch_change(last_direction):
    while record_button.is_active and not exit_requested:
        new_direction = get_translation_direction()
        if new_direction != last_direction:
            return new_direction
        time.sleep(0.01)
    if exit_requested:
        return "exit"
    return None

try:
    current_direction = get_translation_direction()
    load_models(current_direction)
    play_mode_announcement(current_direction)

    while not exit_requested:
        red_led.off()
        green_led.off()
        print(f"Mode: {'ES->EN' if current_direction == 'es_to_en' else 'EN->ES'}")
        print("Press/hold button to record (or toggle switch to change direction).")
        direction_changed = wait_for_button_or_switch_change(current_direction)
        if exit_requested or direction_changed == "exit":
            break
        if direction_changed:
            current_direction = direction_changed
            load_models(current_direction)
            play_mode_announcement(current_direction)
            continue

        q = queue.Queue()
        frames = []
        recording = True

        def callback(indata, frames_count, time_info, status):
            if status:
                print(status, file=sys.stderr)
            if recording:
                q.put(indata.copy())

        try:
            stream = sd.InputStream(
                samplerate=16000,
                channels=1,
                dtype='int16',
                device=INPUT_DEVICE,
                callback=callback
            )
            stream.start()
        except Exception as e:
            print(f"Failed to open input stream on {INPUT_DEVICE} with 16000 Hz: {e}")
            # Optional: try alternate device or sample rate as fallback
        red_led.on()
        print("Recording... (release button to stop, or toggle switch to change direction)")
        direction_changed = None
        try:
            direction_changed = wait_for_button_release_or_switch_change(current_direction)
        finally:
            recording = False
            stream.stop()
            stream.close()
            red_led.off()

        if exit_requested or direction_changed == "exit":
            break
        elif direction_changed:
            current_direction = direction_changed
            load_models(current_direction)
            play_mode_announcement(current_direction)
            continue

        while not q.empty():
            frames.append(q.get())
        audio = np.concatenate(frames, axis=0) if frames else np.array([], dtype='int16')
        if audio.size == 0:
            print("No audio recorded")
            play_no_audio(current_direction)
            continue

        print("Transcribing...")
        red_led.on()
        text = transcribe(audio, 16000)
        red_led.off()
        print(f"Recognized: {text}")

        if not text.strip():
            print("No speech detected")
            play_no_audio(current_direction)
            continue

        # --- PUNCTUATE ---
        print("Restoring punctuation...")
        text = punct_model.restore_punctuation(text)
        print(f"Punctuated: {text}")

        print("Translating...")
        red_led.on()
        translation = translate(text)
        red_led.off()

        green_led.on()
        print(f"Translation: {translation}\nPress button to play translation (or toggle switch to change direction).")
        play_translation = False
        while not exit_requested:
            if record_button.is_active:
                play_translation = True
                break
            new_direction = get_translation_direction()
            if new_direction != current_direction:
                green_led.off()
                current_direction = new_direction
                load_models(current_direction)
                play_mode_announcement(current_direction)
                play_translation = False
                break
            time.sleep(0.01)
        green_led.off()
        if exit_requested:
            break
        if not play_translation:
            continue

        print("Speaking translation...")
        speak(translation)

except KeyboardInterrupt:
    print("\nKeyboard interrupt received")
finally:
    print("\nExiting translator")
    red_led.off()
    green_led.off()
    #sys.exit(0)
    print("Shutting down")
    os.system("sudo shutdown -h now")
