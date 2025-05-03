from flask import Flask, request, jsonify
import logging
import whisper
import tempfile
import os
import soundfile as sf
import wave

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app and Whisper model
app = Flask(__name__)
model = whisper.load_model("tiny")

def is_valid_audio(file_path):
    try:
        # Try with soundfile first
        try:
            data, samplerate = sf.read(file_path)
            return len(data) > 0 and samplerate > 0
        except:
            # If soundfile fails, try with wave
            try:
                with wave.open(file_path, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    return frames > 0 and rate > 0
            except:
                logger.error("File is neither valid WAV nor supported by soundfile")
                return False
    except Exception as e:
        logger.error(f"Audio validation failed: {str(e)}")
        return False

@app.route('/test', methods=['GET'])
def test_get():
    return "Server is running!"

@app.route('/voice_to_text', methods=['POST'])
def voice_to_text():
    try:
        # Log request details
        logger.debug(f"Received request: Content-Type: {request.content_type}")
        logger.debug(f"Request length: {len(request.data)} bytes")

        if not request.data:
            logger.warning("Received empty data")
            return jsonify({"status": "error", "message": "Empty data received"})

        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(request.data)
            temp_filename = temp_file.name

        try:
            # Validate audio file
            if not is_valid_audio(temp_filename):
                return jsonify({
                    "status": "error",
                    "message": "Invalid or corrupted audio file"
                }), 400

            # Transcribe the audio using Whisper
            result = model.transcribe(
                temp_filename,
                fp16=False,
                language='en'
            )
            
            transcribed_text = result["text"].strip()
            
            if not transcribed_text:
                return jsonify({
                    "status": "error",
                    "message": "No speech detected in audio"
                }), 400
            
            logger.info(f"Transcribed text: {transcribed_text}")
            
            return jsonify({
                "status": "success",
                "text": transcribed_text
            })

        finally:
            # Clean up the temporary file
            os.unlink(temp_filename)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
