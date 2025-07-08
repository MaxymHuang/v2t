from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
import logging
import tempfile
import os
import soundfile as sf
from faster_whisper import WhisperModel
import wave

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize FastAPI app and Whisper model
app = FastAPI()
# Initialize faster-whisper model with compute type
model = WhisperModel("medium", device="cpu", compute_type="int8")

def is_valid_audio(file_path):
    try:
        # Try with soundfile first
        try:
            data, samplerate = sf.read(file_path)
            return len(data) > 0 and samplerate > 0
        except Exception:
            # If soundfile fails, try with wave
            try:
                with wave.open(file_path, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    return frames > 0 and rate > 0
            except Exception:
                logger.error("File is neither valid WAV nor supported by soundfile")
                return False
    except Exception as e:
        logger.error(f"Audio validation failed: {str(e)}")
        return False

@app.get("/test")
async def test_get():
    # Return plain text response to mimic original headers as closely as possible
    return PlainTextResponse("Server is running!")

@app.post("/voice_to_text")
async def voice_to_text(request: Request):
    try:
        # Log request details
        content_type = request.headers.get('Content-Type')
        body = await request.body()
        logger.debug(f"Received request: Content-Type: {content_type}")
        logger.debug(f"Request length: {len(body)} bytes")

        if not body:
            logger.warning("Received empty data")
            return JSONResponse(status_code=400, content={"status": "error", "message": "Empty data received"})

        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(body)
            temp_filename = temp_file.name

        try:
            # Validate audio file
            if not is_valid_audio(temp_filename):
                return JSONResponse(status_code=400, content={
                    "status": "error",
                    "message": "Invalid or corrupted audio file"
                })

            # Transcribe the audio using faster-whisper with adjusted parameters
            segments, info = model.transcribe(
                temp_filename,
                language='en',
                temperature=0.0,  # Start with lowest temperature
                beam_size=5,      # Increase beam size for better accuracy
                best_of=2,        # Number of candidates to consider
                condition_on_previous_text=True,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,  # Adjust probability threshold
                no_speech_threshold=0.6
            )
            
            # Combine all segments into one text
            transcribed_text = " ".join([segment.text for segment in segments]).strip()
            
            if not transcribed_text:
                return JSONResponse(status_code=400, content={
                    "status": "error",
                    "message": "No speech detected in audio"
                })
            
            logger.info(f"Transcribed text: {transcribed_text}")
            
            return JSONResponse(content={
                "text": transcribed_text
            })

        finally:
            # Clean up the temporary file
            os.unlink(temp_filename)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting server on http://0.0.0.0:5000")
    uvicorn.run("server:app", host="0.0.0.0", port=5000, reload=True)
