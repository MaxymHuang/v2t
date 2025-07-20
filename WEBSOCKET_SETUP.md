# WebSocket Streaming Voice-to-Text Setup

This document explains how to set up and use the new WebSocket-based streaming voice-to-text system that eliminates the 2-second recording limitation and provides better audio quality at 16kHz.

## Key Improvements

### Audio Quality
- **16kHz Sample Rate**: Increased from 4kHz/8kHz to 16kHz for much better speech recognition
- **Real-time Streaming**: No more 2-second recording limitation
- **Continuous Audio**: Stream audio in 500ms chunks for near real-time processing
- **Better Memory Management**: Smaller buffers, more efficient memory usage

### Technical Benefits
- **Lower Latency**: Start getting transcriptions while still speaking
- **Reduced Memory Usage**: No large audio buffers on ESP32
- **Better Connection Management**: WebSocket with automatic reconnection
- **Dual-core Processing**: Audio streaming runs on dedicated core

## Required Libraries

### ESP32 Libraries (Arduino IDE)
Install these libraries through the Arduino Library Manager:

```
1. WebSockets by Markus Sattler (v2.3.6 or later)
2. ArduinoJson by Benoit Blanchon (v6.19.4 or later)
3. Adafruit GFX Library
4. Adafruit SSD1306
5. WiFi (ESP32 built-in)
```

### Python Server Dependencies
The server dependencies are already updated in `requirements.txt`:

```bash
pip install -r requirements.txt
```

New dependencies added:
- `websockets`
- `python-socketio`

## Hardware Setup

Same hardware setup as before:

| Component | ESP32 Pin |
|-----------|-----------|
| I2S WS    | GPIO 25   |
| I2S SD    | GPIO 33   |
| I2S SCK   | GPIO 26   |
| OLED SDA  | GPIO 21   |
| OLED SCL  | GPIO 22   |
| Button    | GPIO 2    |

## Configuration

### ESP32 Configuration (`esp32_websocket.cpp`)

Update these constants:

```cpp
const char* ssid      = "YOUR_WIFI_SSID";
const char* password  = "YOUR_WIFI_PASSWORD";
const char* serverIP  = "192.168.1.100";  // Your server IP
const int   serverPort = 5000;            // Server port
```

### Audio Quality Settings

The system is optimized for streaming:

```cpp
#define SAMPLE_RATE         16000  // 16kHz for premium quality
#define CHUNK_DURATION_MS   500    // 500ms chunks
#define CHUNK_SIZE          16000  // 16KB per chunk
```

## Usage Instructions

### 1. Start the Server

```bash
python server.py
```

The server will start with both HTTP and WebSocket endpoints:
- HTTP endpoints: `http://localhost:5000/test`, `/voice_to_text`, etc.
- WebSocket endpoint: `ws://localhost:5000/ws/audio/{client_id}`

### 2. Flash ESP32

1. Open `esp32_websocket.cpp` in Arduino IDE
2. Update WiFi credentials and server IP
3. Select your ESP32 board and port
4. Upload the code

### 3. Operation

1. **Power on ESP32**: Device will connect to WiFi and WebSocket
2. **Ready State**: Display shows "Ready to Record" with quality info
3. **Recording**: Hold button to start streaming audio
4. **Real-time Processing**: Audio is processed in 500ms chunks
5. **Results**: Transcription appears on display within seconds

## WebSocket Protocol

### Client to Server Messages

**Start Recording:**
```json
{
  "command": "start_recording",
  "sample_rate": 16000,
  "chunk_size": 16000
}
```

**Stop Recording:**
```json
{
  "command": "stop_recording"
}
```

**Keep-alive Ping:**
```json
{
  "command": "ping"
}
```

**Audio Data:**
Binary WebSocket messages containing 16-bit PCM audio chunks.

### Server to Client Messages

**Status Response:**
```json
{
  "type": "status",
  "message": "Recording started",
  "sample_rate": 16000
}
```

**Transcription Result:**
```json
{
  "type": "transcription",
  "text": "Hello world",
  "timestamp": "2024-01-15T10:30:00.123Z"
}
```

**Error Message:**
```json
{
  "type": "error",
  "message": "Transcription failed: ..."
}
```

## Monitoring and Debugging

### Server Endpoints

**WebSocket Status:**
```bash
curl http://localhost:5000/ws/status
```

**Training Stats:**
```bash
curl http://localhost:5000/training/stats
```

**Audio Diagnostic:**
```bash
curl http://localhost:5000/audio/diagnostic
```

### ESP32 Serial Monitor

Enable serial monitoring at 115200 baud to see:
- Connection status
- Audio chunk transmission
- WebSocket events
- Error messages

### Common Issues and Solutions

**1. WebSocket Connection Failed**
- Check server IP and port
- Ensure server is running
- Verify firewall settings

**2. Audio Quality Issues**
- Confirm 16kHz sample rate
- Check I2S microphone connections
- Monitor serial output for I2S errors

**3. Memory Issues**
- ESP32 should use ~50KB RAM for audio buffers
- If crashes occur, reduce `CHUNK_SIZE` or `DMA_BUFFER_COUNT`

**4. Latency Issues**
- Reduce `CHUNK_DURATION_MS` for lower latency
- Increase for better transcription accuracy

## Performance Comparison

| Feature | HTTP POST (Old) | WebSocket Streaming (New) |
|---------|----------------|---------------------------|
| Sample Rate | 4-8kHz | 16kHz |
| Recording Limit | 2-8 seconds | Unlimited |
| Memory Usage | High (64KB+) | Low (~16KB) |
| Latency | High (2s+ wait) | Low (500ms chunks) |
| Audio Quality | Basic | Premium |
| Connection | One-shot | Persistent |
| Real-time Processing | No | Yes |

## Advanced Configuration

### Adjusting Chunk Size
For different latency/quality trade-offs:

```cpp
// Lower latency (250ms chunks)
#define CHUNK_DURATION_MS   250
#define CHUNK_SIZE          8000

// Higher quality (1000ms chunks)
#define CHUNK_DURATION_MS   1000
#define CHUNK_SIZE          32000
```

### Memory Optimization
For ESP32 with limited RAM:

```cpp
#define DMA_BUFFER_COUNT    4      // Reduce buffers
#define DMA_BUFFER_LEN      256    // Smaller DMA buffers
```

### Sample Rate Adjustment
Server automatically handles different sample rates:
- ESP32 sends at 16kHz
- Server processes at 16kHz
- Whisper model receives 16kHz input

## Troubleshooting

### ESP32 Won't Connect
1. Check WiFi credentials
2. Verify server IP address
3. Ensure server is running and accessible
4. Check serial monitor for connection errors

### Poor Audio Quality
1. Verify I2S microphone is working
2. Check sample rate configuration
3. Monitor audio levels in serial output
4. Test with different microphones

### High Latency
1. Reduce chunk duration
2. Check network connectivity
3. Monitor server processing time
4. Ensure sufficient server resources

## Migration from HTTP Version

To migrate from the HTTP POST version:

1. **Install new libraries** (WebSockets, ArduinoJson)
2. **Update server** with new WebSocket endpoints
3. **Flash new ESP32 code** (`esp32_websocket.cpp`)
4. **Update configuration** (IP, credentials)
5. **Test streaming functionality**

The HTTP endpoints remain available for compatibility. 