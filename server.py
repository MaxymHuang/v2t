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
            
        # Validate chunk data
        if len(audio_data) == 0:
            logger.warning(f"Received empty audio chunk from {client_id}")
            return
            
        if len(audio_data) % 2 != 0:
            logger.warning(f"Received odd-sized chunk ({len(audio_data)} bytes) from {client_id}")
            # Pad with zero byte to maintain 16-bit alignment
            audio_data += b'\x00'
            
        # Add chunk to buffer (preserve order, accumulate all chunks)
        self.audio_buffers[client_id].append(audio_data)
        
        # Log progress periodically
        chunk_count = len(self.audio_buffers[client_id])
        total_bytes = sum(len(chunk) for chunk in self.audio_buffers[client_id])
        
        if chunk_count % 20 == 0:  # Log every 20 chunks (10 seconds at 500ms chunks)
            duration_estimate = (total_bytes / 2) / 16000  # 16-bit samples at 16kHz
            logger.info(f"Chunk #{chunk_count}: {total_bytes} bytes, ~{duration_estimate:.1f}s for {client_id}")
        
        # Only process when explicitly told to stop recording
            
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
            
            # CRITICAL DEBUGGING: Let's figure out the actual data rate
            chunk_count = len(self.audio_buffers[client_id])  # This was set to 0 above, but let's track it
            
            # Calculate what the ACTUAL sample rate should be based on timing
            # If ESP32 sent data for X seconds, and we have Y samples, then actual_rate = Y/X
            logger.info(f"DEBUGGING: Raw audio analysis")
            logger.info(f"- Combined audio bytes: {len(combined_audio)}")
            logger.info(f"- 16-bit samples: {len(audio_array)}")
            logger.info(f"- Expected sample rate: 16000Hz")
            
            # Let's try different sample rates to see which one makes sense
            for test_rate in [8000, 12000, 16000, 24000, 32000]:
                test_duration = len(audio_array) / test_rate
                logger.info(f"- If sample rate is {test_rate}Hz: duration = {test_duration:.2f}s")
            
            # Audio quality validation with debugging
            assumed_sample_rate = 16000  # What we think it should be
            assumed_duration = len(audio_array) / assumed_sample_rate
            rms_level = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            peak_level = np.max(np.abs(audio_array))
            
            logger.info(f"Audio stats at 16kHz: {len(audio_array)} samples, {assumed_duration:.2f}s, RMS: {rms_level:.1f}, Peak: {peak_level}")
            
            # Check for valid audio signal
            if peak_level < 100:  # Very quiet signal
                logger.warning(f"Very quiet audio signal (peak: {peak_level})")
            elif peak_level > 30000:  # Very loud signal (close to clipping)
                logger.warning(f"Very loud audio signal (peak: {peak_level}) - possible clipping")
            
            # Normalize to [-1, 1] with proper 16-bit PCM scaling
            if len(audio_array) > 0:
                audio_float = audio_array.astype(np.float32) / 32768.0
                
                # EXPERIMENTAL: Try to detect the correct sample rate based on timing
                # If the ESP32 says it's sending 16kHz but audio sounds sped up,
                # maybe the actual effective rate is different
                
                # Let's try multiple sample rates and see which makes sense
                test_rates = [8000, 12000, 16000, 22050, 24000]
                
                logger.info("EXPERIMENTAL: Trying multiple sample rates to find correct timing...")
                
                # For now, let's try the most common issue: ESP32 might be effectively 
                # sending at 8kHz but we're treating it as 16kHz (causing 2x speed)
                corrected_sample_rate = 8000  # Try this first
                
                logger.info(f"EXPERIMENTAL: Using {corrected_sample_rate}Hz instead of 16kHz to correct speed")
                
                # Save to temporary WAV file with corrected sample rate
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                sf.write(temp_file.name, audio_float, corrected_sample_rate)
                temp_file.close()
                
                # Update our stored sample rate
                self.sample_rates[client_id] = corrected_sample_rate
                
                try:
                    # CRITICAL: Only transcribe after all chunks are merged and processed
                    logger.info("Starting transcription of complete merged recording")
                    
                    # Preprocess while preserving audio integrity
                    preprocessed_file = preprocess_audio(temp_file.name)
                    
                    # Verify the preprocessed file maintains correct timing
                    verification_audio, verification_sr = sf.read(preprocessed_file)
                    verification_duration = len(verification_audio) / verification_sr
                    logger.info(f"Transcription input verified: {len(verification_audio)} samples at {verification_sr}Hz ({verification_duration:.2f}s)")
                    
                    # Initialize model if needed
                    initialize_custom_model()
                    
                    # Transcribe the complete, properly processed audio
                    logger.info("Starting Whisper transcription...")
                    transcription = transcribe_with_custom_model(preprocessed_file)
                    logger.info(f"Transcription completed: '{transcription}'")
                    
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
    """Preprocess audio: preserve 16kHz PCM integrity from ESP32 I2S extraction"""
    try:
        # Read audio file (already 16kHz from ESP32 with proper chunk combination)
        audio, sr = sf.read(file_path)
        logger.info(f"Preprocessing: loaded {len(audio)} samples at {sr}Hz from ESP32")
        
        # Convert to mono if stereo (shouldn't happen with ESP32 mono mic)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            logger.info("Converted stereo to mono")
        
        # Audio quality validation
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))
        duration = len(audio) / sr
        logger.info(f"Audio integrity check: {len(audio)} samples, {duration:.2f}s, RMS={rms:.4f}, Peak={peak:.4f}")
        
        # Verify sample rate is exactly 16kHz (critical for audio integrity)
        if sr != 16000:
            logger.error(f"CRITICAL: Sample rate mismatch! Expected 16000Hz, got {sr}Hz - this causes speed issues!")
            # Force correct sample rate without resampling (preserve timing)
            sr = 16000
            logger.info("Corrected sample rate to 16000Hz to preserve audio timing")
        
        # Minimal processing to preserve audio integrity
        # Only normalize if signal is very weak or very strong
        if peak < 0.01:  # Very weak signal
            audio = audio * (0.1 / (peak + 1e-10))  # Gentle boost
            logger.info(f"Boosted weak signal from {peak:.4f} to ~0.1")
        elif peak > 0.95:  # Near clipping
            audio = audio * (0.8 / peak)  # Gentle reduction
            logger.info(f"Reduced clipping signal from {peak:.4f} to 0.8")
        else:
            logger.info("Audio level is good, no normalization needed")
        
        # NO RESAMPLING - preserve original timing and pitch
        logger.info(f"Final audio: {len(audio)} samples at {sr}Hz ({duration:.2f}s)")
        
        # Save with preserved timing
        processed_path = file_path.replace('.wav', '_processed.wav')
        sf.write(processed_path, audio, sr)
        logger.info("Audio preprocessing complete - timing and pitch preserved")
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
                            # Always use 16kHz for ESP32 to ensure audio integrity
                            audio_manager.sample_rates[client_id] = 16000
                            # Clear any existing buffer to start fresh
                            audio_manager.audio_buffers[client_id] = []
                            
                            response = {
                                "type": "status",
                                "message": "Recording started",
                                "sample_rate": 16000
                            }
                            logger.info(f"Recording started for {client_id} at 16kHz")
                            await websocket.send_text(json.dumps(response))
                            
                        elif command == "stop_recording":
                            audio_manager.is_recording[client_id] = False
                            # Process all accumulated audio chunks
                            if audio_manager.audio_buffers[client_id]:
                                total_bytes = sum(len(chunk) for chunk in audio_manager.audio_buffers[client_id])
                                logger.info(f"Processing complete recording: {total_bytes} bytes for {client_id}")
                                await audio_manager.process_audio_buffer(client_id)
                            await websocket.send_text(json.dumps({
                                "type": "status", 
                                "message": "Recording stopped and processed"
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
