from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
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
import asyncio
import json
from typing import Dict, List
import io

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

# WebSocket Audio Stream Manager
class AudioStreamManager:
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self.audio_buffers: Dict[str, List[bytes]] = {}
        self.sample_rates: Dict[str, int] = {}
        self.is_recording: Dict[str, bool] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.connections[client_id] = websocket
        self.audio_buffers[client_id] = []
        self.sample_rates[client_id] = 16000  # Default to 16kHz
        self.is_recording[client_id] = False
        logger.info(f"WebSocket client {client_id} connected")
        
    def disconnect(self, client_id: str):
        if client_id in self.connections:
            del self.connections[client_id]
        if client_id in self.audio_buffers:
            del self.audio_buffers[client_id]
        if client_id in self.sample_rates:
            del self.sample_rates[client_id]
        if client_id in self.is_recording:
            del self.is_recording[client_id]
        logger.info(f"WebSocket client {client_id} disconnected")
        
    async def handle_audio_chunk(self, client_id: str, audio_data: bytes):
        """Handle incoming audio chunk from ESP32"""
        if client_id not in self.audio_buffers:
            logger.warning(f"Received audio for unknown client {client_id}")
            return
            
        # Add chunk to buffer
        self.audio_buffers[client_id].append(audio_data)
        
        # If we have enough data (approximately 2 seconds at 16kHz), process it
        total_bytes = sum(len(chunk) for chunk in self.audio_buffers[client_id])
        
        # Estimate: 16kHz * 1 channel * 2 bytes/sample * 2 seconds = 64KB
        # Use smaller buffer for very responsive processing
        if total_bytes >= 16000:  # ~0.5 seconds of 16kHz mono audio for very fast response
            logger.info(f"Processing {total_bytes} bytes for {client_id}")
            await self.process_audio_buffer(client_id)
            
    async def process_audio_buffer(self, client_id: str):
        """Process accumulated audio buffer and transcribe"""
        if client_id not in self.audio_buffers or not self.audio_buffers[client_id]:
            return
            
        try:
            # Combine all chunks
            combined_audio = b''.join(self.audio_buffers[client_id])
            
            # Clear buffer for next chunk
            self.audio_buffers[client_id] = []
            
            logger.info(f"Processing audio buffer: {len(combined_audio)} bytes for {client_id}")
            
            # Validate audio data size (must be even for 16-bit samples)
            if len(combined_audio) % 2 != 0:
                logger.warning(f"Invalid audio data size: {len(combined_audio)} bytes (not divisible by 2)")
                # Pad with one zero byte if odd length
                combined_audio += b'\x00'
            
            # Convert to numpy array (16-bit PCM from ESP32 I2S extraction)
            audio_array = np.frombuffer(combined_audio, dtype=np.int16)
            
            # Audio quality validation
            sample_rate = self.sample_rates[client_id]
            duration = len(audio_array) / sample_rate
            rms_level = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            peak_level = np.max(np.abs(audio_array))
            
            logger.info(f"Audio stats: {len(audio_array)} samples, {duration:.2f}s, RMS: {rms_level:.1f}, Peak: {peak_level}")
            
            # Check for valid audio signal
            if peak_level < 100:  # Very quiet signal
                logger.warning(f"Very quiet audio signal (peak: {peak_level})")
            elif peak_level > 30000:  # Very loud signal (close to clipping)
                logger.warning(f"Very loud audio signal (peak: {peak_level}) - possible clipping")
            
            # Normalize to [-1, 1] with proper 16-bit PCM scaling
            if len(audio_array) > 0:
                audio_float = audio_array.astype(np.float32) / 32768.0
                
                # Save to temporary WAV file for transcription
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                sf.write(temp_file.name, audio_float, self.sample_rates[client_id])
                temp_file.close()
                
                try:
                    # Preprocess and transcribe
                    preprocessed_file = preprocess_audio(temp_file.name)
                    
                    # Initialize model if needed
                    initialize_custom_model()
                    
                    # Transcribe
                    transcription = transcribe_with_custom_model(preprocessed_file)
                    
                    # Send result back to client
                    if client_id in self.connections:
                        await self.connections[client_id].send_text(json.dumps({
                            "type": "transcription",
                            "text": transcription,
                            "timestamp": datetime.datetime.now().isoformat()
                        }))
                        
                    # Save for training
                    if not transcription.startswith("ERROR:") and transcription:
                        save_audio_for_training(preprocessed_file, transcription, self.sample_rates[client_id])
                        
                    # Cleanup
                    os.unlink(preprocessed_file)
                    
                except Exception as e:
                    logger.error(f"Transcription error for client {client_id}: {e}")
                    if client_id in self.connections:
                        await self.connections[client_id].send_text(json.dumps({
                            "type": "error",
                            "message": f"Transcription failed: {str(e)}"
                        }))
                finally:
                    os.unlink(temp_file.name)
                    
        except Exception as e:
            logger.error(f"Audio processing error for client {client_id}: {e}")

# Global audio stream manager
audio_manager = AudioStreamManager()

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
    """Preprocess audio: handle 16kHz PCM from ESP32 I2S extraction"""
    import scipy.signal
    try:
        # Read audio file (should already be 16kHz from ESP32)
        audio, sr = sf.read(file_path)
        logger.info(f"Preprocessing: loaded {len(audio)} samples at {sr}Hz from ESP32")
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            logger.info("Converted stereo to mono")
        
        # Audio quality check
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))
        logger.info(f"Audio quality: RMS={rms:.4f}, Peak={peak:.4f}")
        
        # Gentle normalization (preserve dynamics, avoid over-normalization)
        if peak > 0.1:  # Only normalize if signal is strong enough
            # Normalize to 80% of full scale to avoid clipping
            audio = audio * (0.8 / peak)
            logger.info(f"Normalized audio to 80% scale (was {peak:.4f})")
        else:
            logger.warning(f"Weak audio signal (peak: {peak:.4f}), minimal processing")
        
        # Ensure we have 16kHz (ESP32 should already provide this)
        target_sr = 16000
        if abs(sr - target_sr) > 10:  # Allow small tolerance
            logger.info(f"Resampling from {sr}Hz to {target_sr}Hz")
            duration = len(audio) / sr
            num_samples = int(duration * target_sr)
            audio = scipy.signal.resample(audio, num_samples)
            sr = target_sr
        else:
            logger.info(f"Sample rate {sr}Hz is correct, no resampling needed")
        
        # For streaming: use shorter buffer (2 seconds max for responsive processing)
        target_length = min(2 * target_sr, len(audio))  # Max 2 seconds
        if len(audio) > target_length:
            audio = audio[:target_length]
            logger.info(f"Trimmed to {target_length} samples ({target_length/target_sr:.1f}s)")
        elif len(audio) < target_sr * 0.1:  # Less than 100ms
            logger.warning(f"Very short audio: {len(audio)} samples ({len(audio)/target_sr:.3f}s)")
        
        # Save preprocessed audio
        processed_path = file_path.replace('.wav', '_processed.wav')
        sf.write(processed_path, audio, target_sr)
        logger.info(f"Saved preprocessed audio: {len(audio)} samples at {target_sr}Hz")
        return processed_path
        
    except Exception as e:
        logger.error(f"Audio preprocessing failed: {e}", exc_info=True)
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
        
        # Enhanced audio analysis for ESP32 I2S debugging
        rms_level = np.sqrt(np.mean(audio_data ** 2))
        peak_level = np.max(np.abs(audio_data))
        dynamic_range = peak_level / (rms_level + 1e-10)  # Avoid division by zero
        
        # Check for common I2S extraction issues
        zero_samples = np.sum(audio_data == 0)
        clipped_samples = np.sum(np.abs(audio_data) > 0.95)
        
        # Get corresponding metadata
        metadata_file = latest_file.replace('.wav', '.txt')
        metadata_path = os.path.join(TRAINING_DIR, metadata_file)
        transcription = "No transcription found"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if "Transcription:" in content:
                    transcription = content.split("Transcription:")[1].split("\n")[0].strip()
        
        # Diagnose potential issues
        issues = []
        if peak_level < 0.01:
            issues.append("Very quiet signal - check ESP32 microphone connection")
        if clipped_samples > len(audio_data) * 0.01:  # More than 1% clipped
            issues.append("Audio clipping detected - reduce microphone gain")
        if zero_samples > len(audio_data) * 0.1:  # More than 10% zeros
            issues.append("Many zero samples - possible I2S extraction issue")
        if dynamic_range < 2:
            issues.append("Low dynamic range - audio may be compressed")
        
        return JSONResponse(content={
            "latest_audio_file": latest_file,
            "sample_rate_hz": sr,
            "duration_seconds": round(duration, 2),
            "audio_length_samples": len(audio_data),
            "transcription": transcription,
            "audio_quality": {
                "rms_level": round(rms_level, 4),
                "peak_level": round(peak_level, 4),
                "dynamic_range": round(dynamic_range, 2),
                "zero_samples": zero_samples,
                "clipped_samples": clipped_samples,
                "zero_percentage": round(100 * zero_samples / len(audio_data), 2),
                "clipped_percentage": round(100 * clipped_samples / len(audio_data), 2)
            },
            "esp32_analysis": {
                "i2s_extraction": "16-bit PCM from 32-bit I2S words",
                "expected_sample_rate": "16000 Hz",
                "sample_rate_status": "✓ Correct" if sr == 16000 else f"⚠ Unexpected: {sr} Hz",
                "audio_quality": "Premium" if sr == 16000 and peak_level > 0.01 else "Needs attention",
                "potential_issues": issues if issues else ["None detected"]
            },
            "diagnostic_info": {
                "esp32_sample_rate": sr,
                "server_processing": f"{sr} Hz from ESP32 I2S extraction",
                "pitch_preservation": "✓ Original pitch maintained" if sr == 16000 else "⚠ Pitch may be altered",
                "quality_level": "Premium (16kHz)" if sr == 16000 else f"Unexpected ({sr} Hz)"
            }
        })
        
    except Exception as e:
        logger.error(f"Error in audio diagnostic: {e}")
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": f"Failed to get audio diagnostic: {str(e)}"
        })

@app.websocket("/ws/audio/{client_id}")
async def websocket_audio_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time audio streaming"""
    await audio_manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from ESP32
            try:
                message = await websocket.receive()
            except Exception as e:
                logger.error(f"Error receiving message from {client_id}: {e}")
                break
            
            if message["type"] == "websocket.receive":
                if "bytes" in message:
                    # Handle binary audio data
                    audio_data = message["bytes"]
                    logger.debug(f"Received {len(audio_data)} bytes from {client_id}")
                    await audio_manager.handle_audio_chunk(client_id, audio_data)
                    
                elif "text" in message:
                    # Handle text commands
                    try:
                        data = json.loads(message["text"])
                        command = data.get("command", "")
                        
                        if command == "start_recording":
                            logger.info(f"Start recording command from {client_id}")
                            audio_manager.is_recording[client_id] = True
                            audio_manager.sample_rates[client_id] = data.get("sample_rate", 16000)
                            
                            response = {
                                "type": "status",
                                "message": "Recording started",
                                "sample_rate": audio_manager.sample_rates[client_id]
                            }
                            logger.info(f"Sending response to {client_id}: {response}")
                            await websocket.send_text(json.dumps(response))
                            
                        elif command == "stop_recording":
                            audio_manager.is_recording[client_id] = False
                            # Process any remaining audio in buffer
                            if audio_manager.audio_buffers[client_id]:
                                await audio_manager.process_audio_buffer(client_id)
                            await websocket.send_text(json.dumps({
                                "type": "status",
                                "message": "Recording stopped"
                            }))
                            
                        elif command == "ping":
                            await websocket.send_text(json.dumps({
                                "type": "pong",
                                "timestamp": datetime.datetime.now().isoformat()
                            }))
                            
                    except json.JSONDecodeError:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "Invalid JSON format"
                        }))
                        
    except WebSocketDisconnect:
        logger.info(f"WebSocket client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
    finally:
        audio_manager.disconnect(client_id)

@app.get("/ws/status")
async def websocket_status():
    """Get WebSocket connection status"""
    return JSONResponse(content={
        "active_connections": len(audio_manager.connections),
        "clients": list(audio_manager.connections.keys()),
        "buffer_status": {
            client_id: {
                "buffer_size": len(audio_manager.audio_buffers.get(client_id, [])),
                "sample_rate": audio_manager.sample_rates.get(client_id, 0),
                "is_recording": audio_manager.is_recording.get(client_id, False)
            }
            for client_id in audio_manager.connections.keys()
        }
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
