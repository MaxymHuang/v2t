# ESP32 Voice-to-Text Project

A voice recognition system using ESP32 microcontroller with I2S microphone that transcribes speech to text using Whisper AI.

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
| Button    | GPIO 0    |

## Software Requirements

### Server Side
- Python 3.11+
- Flask
- OpenAI Whisper
- FFmpeg
- SoundFile

### Client Side
- Arduino IDE
- ESP32 Board Support Package
- Required Libraries:
  - WiFi
  - HTTPClient
  - Wire
  - Adafruit_GFX
  - Adafruit_SSD1306

## Setup Instructions

1. Server Setup:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure ESP32:
   - Update WiFi credentials in esp32.cpp
   - Set correct server IP and port
   - Flash the code to ESP32

3. Start the Server:
   ```bash
   python server.py
   ```

## Usage

1. Press and hold the button to start recording
2. Release the button to stop recording
3. Wait for transcription result on OLED display

## Project Structure

```
v2t/
├── server.py          # Flask server with Whisper AI
├── esp32.cpp          # ESP32 firmware
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Contributing

Feel free to submit issues and pull requests.

## License

MIT License
