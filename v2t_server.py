from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import os

# Initialize Flask app
app = Flask(__name__)

# Load Whisper model (faster implementation)
model = WhisperModel("base", device="cpu", compute_type="int8")

@app.route('/test', methods=['GET'])
def test():
    return "Server is running!"

@app.route('/voice_to_text', methods=['POST'])
def voice_to_text():
    try:
        # Save received audio
        audio_data = request.data
        print(f"Received {len(audio_data)} bytes of audio data")

        # Save raw audio to WAV for Whisper
        with open("received_audio.raw", "wb") as f:
            f.write(audio_data)

        # Convert raw audio to format compatible with Whisper
        import wave
        with wave.open("received_audio.wav", "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)  # 16kHz
            wav_file.writeframes(audio_data)

        # Transcribe using faster-whisper
        segments, info = model.transcribe("received_audio.wav", beam_size=5)

        # Extract the transcribed text
        text = " ".join([segment.text for segment in segments])
        print(f"Transcribed: {text}")

        return jsonify({"text": text})

    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run on all interfaces to accept external connections
    print("Starting server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
