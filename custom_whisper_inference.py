"""
Custom Whisper Model Inference for ESP32 Server
Replace OpenAI Whisper with your custom trained model
"""

import torch
import librosa
import soundfile as sf
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import warnings
warnings.filterwarnings("ignore")

class CustomWhisperModel:
    def __init__(self, model_path="./custom_model"):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.load_model()
    
    def load_model(self):
        """Load custom trained model"""
        try:
            print(f"Loading custom model from {self.model_path}")
            
            # Load processor
            self.processor = WhisperProcessor.from_pretrained(
                self.model_path,
                language="english",
                task="transcribe"
            )
            
            # Load model
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Configure for inference
            self.model.config.forced_decoder_ids = None
            self.model.config.suppress_tokens = []
            self.model.config.use_cache = True
            self.model.eval()
            
            print(f"Custom model loaded successfully")
            print(f"Device: {self.device}")
            
            if torch.cuda.is_available():
                memory_mb = torch.cuda.memory_allocated(0) / 1e6
                print(f"GPU Memory: {memory_mb:.0f} MB")
            
            return True
            
        except Exception as e:
            print(f"Error loading custom model: {e}")
            return False
    
    def transcribe_audio(self, audio_data, sample_rate=16000):
        """Transcribe audio data to text"""
        if self.model is None:
            return "ERROR: Model not loaded"
        
        try:
            # Ensure audio is mono and correct sample rate
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Handle different ESP32 sample rates properly
            if sample_rate == 4000:
                # For ESP32 4000 Hz recordings, use proper resampling
                audio_data = librosa.resample(audio_data, orig_sr=4000, target_sr=16000)
                print(f"Resampled 4000 Hz → 16000 Hz for ESP32 audio")
            elif sample_rate == 6000:
                # For ESP32 6000 Hz recordings, use proper resampling
                audio_data = librosa.resample(audio_data, orig_sr=6000, target_sr=16000)
                print(f"Resampled 6000 Hz → 16000 Hz for ESP32 audio")
            elif sample_rate == 8000:
                # For ESP32 8000 Hz recordings, use proper resampling
                audio_data = librosa.resample(audio_data, orig_sr=8000, target_sr=16000)
                print(f"Resampled 8000 Hz → 16000 Hz for ESP32 audio")
            elif sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            
            # Preprocess
            inputs = self.processor(audio_data, sampling_rate=16000, return_tensors="pt")
            
            # Add attention mask to avoid warnings
            if "attention_mask" not in inputs:
                input_features = inputs["input_features"]
                attention_mask = torch.ones(input_features.shape[0], input_features.shape[1], dtype=torch.long)
                inputs["attention_mask"] = attention_mask
            
            # Move to device
            if self.device == "cuda":
                inputs = {k: v.to(self.device, dtype=torch.float32) for k, v in inputs.items()}
            else:
                inputs = {k: v.to(dtype=torch.float32) for k, v in inputs.items()}
            
            # Generate transcription
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs["input_features"],
                    max_new_tokens=50,
                    do_sample=False,
                    num_beams=1,
                    language="en",
                    task="transcribe",
                    # Explicitly set generation config to avoid warnings
                    generation_config=self.model.generation_config,
                    # Suppress specific tokens to avoid warnings
                    suppress_tokens=[],
                    begin_suppress_tokens=[220, 50257],
                )
            
            # Decode
            transcription = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            return transcription.strip()
            
        except Exception as e:
            return f"ERROR: {e}"

# Global model instance
custom_model = None

def get_custom_model():
    """Get or create custom model instance"""
    global custom_model
    if custom_model is None:
        custom_model = CustomWhisperModel()
    return custom_model

def transcribe_with_custom_model(audio_file_path):
    """Transcribe audio file using custom model"""
    model = get_custom_model()
    
    try:
        # Load audio
        audio, sr = librosa.load(audio_file_path, sr=16000, mono=True)
        
        # Transcribe
        result = model.transcribe_audio(audio, sr)
        
        return result
        
    except Exception as e:
        return f"ERROR: {e}"
