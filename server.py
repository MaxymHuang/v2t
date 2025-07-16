from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
import logging
import tempfile
import os
import soundfile as sf
from custom_whisper_inference import transcribe_with_custom_model
import wave
import numpy as np
import time
import datetime
import shutil

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Suppress verbose debug messages
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('librosa').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)

# Initialize FastAPI app and Whisper model
app = FastAPI()

# CPU-optimized configuration for i7-8700
import os
os.environ["OMP_NUM_THREADS"] = "6"  # Use 6 cores for OpenMP
os.environ["MKL_NUM_THREADS"] = "6"  # Intel MKL optimization

# Initialize custom Whisper model
# The custom model will be loaded when first used
custom_model_initialized = False

# Create training directory if it doesn't exist
TRAINING_DIR = "training"
if not os.path.exists(TRAINING_DIR):
    os.makedirs(TRAINING_DIR)
    logger.info(f"Created training directory: {TRAINING_DIR}")

def save_audio_for_training(audio_file_path, transcribed_text, sample_rate):
    """Save audio file to training directory with metadata"""
    try:
        # Generate timestamp for unique filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        # Create filename with timestamp
        base_filename = f"audio_{timestamp}.wav"
        training_file_path = os.path.join(TRAINING_DIR, base_filename)
        
        # Copy the audio file to training directory
        shutil.copy2(audio_file_path, training_file_path)
        
        # Create metadata file with transcription
        metadata_file_path = training_file_path.replace('.wav', '.txt')
        with open(metadata_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Transcription: {transcribed_text}\n")
            f.write(f"Sample Rate: {sample_rate} Hz\n")
            f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Original File: {os.path.basename(audio_file_path)}\n")
        
        logger.info(f"Saved training audio: {base_filename}")
        logger.info(f"Saved metadata: {os.path.basename(metadata_file_path)}")
        
        return training_file_path
        
    except Exception as e:
        logger.error(f"Failed to save training audio: {e}")
        return None

def initialize_custom_model():
    """Initialize the custom Whisper model on first use"""
    global custom_model_initialized
    if not custom_model_initialized:
        logger.info("Initializing custom Whisper model...")
        try:
            # Test the custom model to ensure it loads properly
            test_result = transcribe_with_custom_model("test_audio.wav")  # This will fail but tests loading
            logger.info("Custom Whisper model initialized successfully")
            custom_model_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize custom model: {e}")
            custom_model_initialized = False

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

def preprocess_audio(file_path):
    """Preprocess audio: resample to 8000 Hz and ensure exactly 4 seconds (32000 samples)"""
    import scipy.signal
    try:
        # Read audio file
        audio, sr = sf.read(file_path)
        print(f"Loaded audio with sample rate: {sr}")
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        # Normalize audio to [-1, 1]
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        # Resample to 8000 Hz if needed
        target_sr = 8000
        if sr != target_sr:
            duration = len(audio) / sr
            num_samples = int(duration * target_sr)
            audio = scipy.signal.resample(audio, num_samples)
            sr = target_sr
        # Ensure exactly 4 seconds (32000 samples at 8000 Hz)
        target_length = 4 * target_sr
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        # Save preprocessed audio
        processed_path = file_path.replace('.wav', '_processed.wav')
        sf.write(processed_path, audio, target_sr)
        return processed_path
    except Exception as e:
        logger.warning(f"Audio preprocessing failed: {e}, using original")
        return file_path

@app.get("/test")
async def test_get():
    # Return plain text response to mimic original headers as closely as possible
    return PlainTextResponse("Server is running!")

@app.get("/training/stats")
async def get_training_stats():
    """Get statistics about saved training data"""
    try:
        if not os.path.exists(TRAINING_DIR):
            return JSONResponse(content={
                "training_files": 0,
                "total_duration": 0,
                "latest_file": None,
                "message": "No training data collected yet"
            })
        
        # Count audio files
        audio_files = [f for f in os.listdir(TRAINING_DIR) if f.endswith('.wav')]
        metadata_files = [f for f in os.listdir(TRAINING_DIR) if f.endswith('.txt')]
        
        # Calculate total duration
        total_duration = 0
        for audio_file in audio_files:
            try:
                audio_path = os.path.join(TRAINING_DIR, audio_file)
                audio_data, sr = sf.read(audio_path)
                duration = len(audio_data) / sr
                total_duration += duration
            except Exception as e:
                logger.warning(f"Could not read duration for {audio_file}: {e}")
        
        # Get latest file
        latest_file = None
        if audio_files:
            latest_file = max(audio_files, key=lambda x: os.path.getctime(os.path.join(TRAINING_DIR, x)))
        
        return JSONResponse(content={
            "training_files": len(audio_files),
            "metadata_files": len(metadata_files),
            "total_duration_seconds": round(total_duration, 2),
            "latest_file": latest_file,
            "training_directory": TRAINING_DIR
        })
        
    except Exception as e:
        logger.error(f"Error getting training stats: {e}")
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": f"Failed to get training stats: {str(e)}"
        })

@app.get("/audio/diagnostic")
async def audio_diagnostic():
    """Get diagnostic information about the latest audio processing"""
    try:
        if not os.path.exists(TRAINING_DIR):
            return JSONResponse(content={
                "message": "No training data available for diagnosis"
            })
        
        # Get the latest audio file
        audio_files = [f for f in os.listdir(TRAINING_DIR) if f.endswith('.wav')]
        if not audio_files:
            return JSONResponse(content={
                "message": "No audio files found for diagnosis"
            })
        
        latest_file = max(audio_files, key=lambda x: os.path.getctime(os.path.join(TRAINING_DIR, x)))
        audio_path = os.path.join(TRAINING_DIR, latest_file)
        
        # Read audio properties
        audio_data, sr = sf.read(audio_path)
        duration = len(audio_data) / sr
        
        # Get corresponding metadata
        metadata_file = latest_file.replace('.wav', '.txt')
        metadata_path = os.path.join(TRAINING_DIR, metadata_file)
        transcription = "No transcription found"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if "Transcription:" in content:
                    transcription = content.split("Transcription:")[1].split("\n")[0].strip()
        
        return JSONResponse(content={
            "latest_audio_file": latest_file,
            "sample_rate_hz": sr,
            "duration_seconds": round(duration, 2),
            "audio_length_samples": len(audio_data),
            "transcription": transcription,
            "diagnostic_info": {
                "esp32_sample_rate": sr,
                "server_processing": f"{sr} Hz preserved" if sr in [4000, 6000, 8000] else f"Upsampled to {sr} Hz",
                "pitch_preservation": "Original pitch maintained" if sr in [4000, 6000, 8000] else "Pitch may be altered",
                "quality_level": "Excellent (8000 Hz)" if sr == 8000 else "Optimal (6000 Hz)" if sr == 6000 else "Good (4000 Hz)" if sr == 4000 else "Other"
            }
        })
        
    except Exception as e:
        logger.error(f"Error in audio diagnostic: {e}")
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": f"Failed to get audio diagnostic: {str(e)}"
        })

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

            # Preprocess audio and get audio info
            preprocessed_filename = preprocess_audio(temp_filename)
            
            # Get audio duration for logging
            audio_data, sample_rate = sf.read(preprocessed_filename)
            audio_duration = len(audio_data) / sample_rate

            # Initialize custom model if needed
            initialize_custom_model()

            # Transcribe the audio using custom Whisper model
            start_time = time.time()
            transcribed_text = transcribe_with_custom_model(preprocessed_filename)
            
            # Check if transcription was successful
            if transcribed_text.startswith("ERROR:"):
                logger.error(f"Custom model transcription failed: {transcribed_text}")
                return JSONResponse(status_code=500, content={
                    "status": "error",
                    "message": f"Transcription failed: {transcribed_text}"
                })
            
            if not transcribed_text:
                return JSONResponse(status_code=400, content={
                    "status": "error",
                    "message": "No speech detected in audio"
                })
            
            logger.info(f"Custom model transcribed text: {transcribed_text}")
            
            processing_time = time.time() - start_time
            logger.info(f"Custom model processing time: {processing_time:.2f}s for {audio_duration:.1f}s audio")

            # Save audio file for training
            training_file_path = save_audio_for_training(
                preprocessed_filename, 
                transcribed_text, 
                sample_rate
            )
            
            if training_file_path:
                logger.info(f"Audio saved for training: {os.path.basename(training_file_path)}")

            return JSONResponse(content={
                "text": transcribed_text
            })

        finally:
            # Clean up the temporary file
            os.unlink(temp_filename)
            # Clean up the preprocessed file
            if os.path.exists(preprocessed_filename):
                os.unlink(preprocessed_filename)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting server on http://0.0.0.0:5000")
    uvicorn.run("server:app", host="0.0.0.0", port=5000, reload=True)
