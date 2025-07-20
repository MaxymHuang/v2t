# ESP32 Voice-to-Text Project

A voice recognition system using ESP32 microcontroller with I2S microphone that transcribes speech to text using Whisper AI. Features premium 16kHz audio quality with optimized I2S processing.

## Hardware Requirements

- ESP32 Development Board (WROVER recommended for PSRAM)
- I2S MEMS Microphone (INMP441 or similar)
- OLED Display (SSD1306 128x64)
- Push Button
- Breadboard and jumper wires

## Pin Connections

| Component | ESP32 Pin |
|-----------|-----------|
| I2S WS    | GPIO 25   |
| I2S SD    | GPIO 33   |
| I2S SCK   | GPIO 26   |
| OLED SDA  | GPIO 21   |
| OLED SCL  | GPIO 22   |
| Button    | GPIO 2    |

**Note:** Button pin changed from GPIO 0 to GPIO 2 to avoid BOOT pin conflicts.

## Audio Quality Features

- **16kHz Premium Sampling Rate**: Optimized for high-quality voice transcription
- **MSB I2S Format**: Enhanced compatibility over standard I2S format
- **APLL Clock Source**: Precise timing for audio integrity
- **Optimized Bit Shifting**: Set to 11-bit shift for improved audio quality from 32-bit I2S to 16-bit PCM conversion
- **Real-time WebSocket Streaming**: 500ms chunks for responsive transcription

## Software Requirements

### Server Side
- Python 3.11+
- FastAPI (or Flask for development)
- OpenAI Whisper
- FFmpeg
- SoundFile
- WebSocket support

### Client Side
- Arduino IDE
- ESP32 Board Support Package
- Required Libraries:
  - WiFi
  - WebSocketsClient
  - ArduinoJson
  - Wire
  - Adafruit_GFX
  - Adafruit_SSD1306
  - I2S driver

## Setup Instructions

### Development Setup

1. **Server Setup:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure ESP32:**
   - Update WiFi credentials in esp32.cpp
   - Set correct server IP and port  
   - Flash the code to ESP32

3. **Start Development Server:**
   ```bash
   python server.py
   ```

### Production Deployment

For production deployment with Docker:

```bash
# Using Docker Compose (recommended)
docker-compose -f docker-compose.production.yml up -d

# Check deployment health
curl http://localhost:5000/health
```

See `README_PRODUCTION.md` for complete production deployment guide.

## Recent Improvements

### Audio Quality Enhancements
- **Bit Shift Optimization**: Changed from 8-bit to 11-bit shift for improved audio quality in I2S to PCM conversion
- **Timing Precision**: APLL clock source ensures accurate 16kHz sampling
- **Format Optimization**: MSB I2S format provides better microphone compatibility

### System Improvements  
- **WebSocket Streaming**: Real-time audio processing with 500ms chunks
- **Production Ready**: Separate production server without development overhead
- **Docker Support**: Full containerization for easy deployment
- **Custom Model Support**: Integration with custom-trained Whisper models

## Usage

1. Press and hold the button to start recording
2. Release the button to stop recording  
3. Wait for transcription result on OLED display
4. Audio is streamed in real-time via WebSocket for faster processing

## Project Structure

```
v2t/
├── server.py                      # Development server with training features
├── server_production.py           # Production server (optimized, no training)
├── esp32.cpp                     # ESP32 firmware with 16kHz I2S processing
├── custom_whisper_inference.py   # Custom Whisper model integration
├── custom_model/                 # Custom trained model files
├── training/                     # Collected training data (dev only)
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Development Docker setup
├── Dockerfile.production         # Production Docker setup
├── docker-compose.production.yml # Production deployment
├── README_PRODUCTION.md          # Production deployment guide
├── WEBSOCKET_SETUP.md           # WebSocket configuration guide
└── README.md                    # This file
```

## Contributing

Feel free to submit issues and pull requests.

## License

MIT License
