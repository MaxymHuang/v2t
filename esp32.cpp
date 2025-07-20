#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <driver/i2s.h>

// Wi-Fi
const char* ssid      = "WIFI_SSID";
const char* password  = "WIFI_PASSWORD";
const char* serverIP  = "SERVER_IP";  // e.g., "192.168.1.100"
const int   serverPort = 5000;

// OLED
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// I2S Mic
#define I2S_WS   25
#define I2S_SD   33
#define I2S_SCK  26

// Button - Use a different GPIO pin to avoid BOOT pin issues
#define BUTTON_PIN 2  // Changed from GPIO 0 to GPIO 2

// Audio configuration for 16kHz streaming
#define SAMPLE_RATE         16000  // 16kHz for premium quality
#define CHUNK_DURATION_MS   500    // 500ms chunks for responsive streaming
#define CHUNK_SIZE          (SAMPLE_RATE * CHUNK_DURATION_MS / 1000 * 4)  // 32KB for 32-bit I2S data
#define DMA_BUFFER_COUNT    8
#define DMA_BUFFER_LEN      512

// Experimental: Different bit shift amounts to try
#define BIT_SHIFT_AMOUNT    8   // Try 8 instead of 11 (less aggressive shift)
// Other values to try: 6, 8, 10, 11, 12, 14, 16

// WebSocket
WebSocketsClient webSocket;
String clientId = "esp32_device";

// Button handling
bool     lastRawState     = HIGH;
uint32_t lastDebounceTime = 0;
const uint32_t debounceDelay = 50;
bool     buttonPressed    = false;
bool     wasPressed       = false;

// State machine
enum State { 
  CONNECTING_WIFI, 
  CONNECTING_WEBSOCKET, 
  IDLE, 
  RECORDING, 
  SHOW_RESULT,
  ERROR_STATE 
};
State state = CONNECTING_WIFI;
uint32_t stateTimestamp = 0;
String   lastTranscription = "";
String   statusMessage = "";

// Audio streaming
uint8_t audioChunk[CHUNK_SIZE];
bool isStreamingAudio = false;
uint32_t lastChunkTime = 0;
uint32_t recordingStartTime = 0;
int totalChunksSent = 0;

// Connection management
bool webSocketConnected = false;
uint32_t lastReconnectAttempt = 0;
const uint32_t RECONNECT_INTERVAL = 5000;

void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  
  Serial.println("=== ESP32 WebSocket Voice Recorder ===");
  Serial.println("Sample Rate: 16kHz");
  Serial.println("Chunk Size: 500ms");

  // OLED init
  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("OLED initialization failed!");
    while(true);
  }
  
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0,0);
  display.println("Starting...");
  display.println("Sample Rate: 16kHz");
  display.display();

  // Initialize I2S for 16kHz streaming
  setupI2S();
  
  // Start WiFi connection
  startWiFiConnection();
}

void setupI2S() {
  // Try a different I2S configuration that might preserve timing better
  i2s_config_t i2s_cfg = {
    .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate          = SAMPLE_RATE,
    .bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_MSB,  // Try MSB format instead
    .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count        = DMA_BUFFER_COUNT,
    .dma_buf_len          = DMA_BUFFER_LEN,
    .use_apll             = true,   // Use APLL for more precise timing
    .tx_desc_auto_clear   = false,
    .fixed_mclk           = 0
  };
  
  i2s_pin_config_t pin_cfg = {
    .bck_io_num   = I2S_SCK,
    .ws_io_num    = I2S_WS,
    .data_out_num = -1,
    .data_in_num  = I2S_SD
  };
  
  esp_err_t result = i2s_driver_install(I2S_NUM_0, &i2s_cfg, 0, nullptr);
  if (result != ESP_OK) {
    Serial.printf("I2S driver install failed: %d\n", result);
    return;
  }
  
  result = i2s_set_pin(I2S_NUM_0, &pin_cfg);
  if (result != ESP_OK) {
    Serial.printf("I2S pin config failed: %d\n", result);
    return;
  }
  
  i2s_zero_dma_buffer(I2S_NUM_0);
  
  // Test actual I2S configuration
  i2s_get_clk(I2S_NUM_0, &result);  // This might help verify the actual clock
  
  Serial.printf("I2S initialized: %dkHz, 32-bit MSB mode with APLL\n", SAMPLE_RATE/1000);
  Serial.printf("Communication format: MSB (changed from standard I2S)\n");
  Serial.printf("APLL enabled for precise timing\n");
}

void startWiFiConnection() {
  Serial.printf("Connecting to WiFi: %s\n", ssid);
  WiFi.begin(ssid, password);
  state = CONNECTING_WIFI;
  stateTimestamp = millis();
}

void setupWebSocket() {
  String wsPath = "/ws/audio/" + clientId;
  webSocket.begin(serverIP, serverPort, wsPath);
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(RECONNECT_INTERVAL);
  webSocket.enableHeartbeat(15000, 3000, 2);
  
  Serial.printf("WebSocket connecting to: ws://%s:%d%s\n", serverIP, serverPort, wsPath.c_str());
}

void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_DISCONNECTED:
      Serial.println("WebSocket Disconnected");
      webSocketConnected = false;
      if (state == RECORDING) {
        stopRecording();
      }
      break;
      
    case WStype_CONNECTED:
      Serial.printf("WebSocket Connected to: %s\n", payload);
      webSocketConnected = true;
      sendPing();
      break;
      
    case WStype_TEXT:
      handleWebSocketMessage((char*)payload);
      break;
      
    case WStype_BIN:
      Serial.printf("Received binary data: %u bytes\n", length);
      break;
      
    case WStype_ERROR:
      Serial.printf("WebSocket Error: %s\n", payload);
      webSocketConnected = false;
      break;
      
    case WStype_FRAGMENT_TEXT_START:
    case WStype_FRAGMENT_BIN_START:
    case WStype_FRAGMENT:
    case WStype_FRAGMENT_FIN:
      break;
  }
}

void handleWebSocketMessage(const char* message) {
  Serial.printf("WebSocket message: %s\n", message);
  
  StaticJsonDocument<1024> doc;
  DeserializationError error = deserializeJson(doc, message);
  
  if (error) {
    Serial.printf("JSON parse error: %s\n", error.c_str());
    return;
  }
  
  String type = doc["type"];
  
  if (type == "transcription") {
    String text = doc["text"];
    lastTranscription = text;
    Serial.printf("Transcription received: %s\n", text.c_str());
    
    // Show result
    state = SHOW_RESULT;
    stateTimestamp = millis();
    
  } else if (type == "status") {
    String message = doc["message"];
    statusMessage = message;
    Serial.printf("Status: %s\n", message.c_str());
    
  } else if (type == "error") {
    String errorMsg = doc["message"];
    Serial.printf("Server error: %s\n", errorMsg.c_str());
    statusMessage = "Error: " + errorMsg;
    
  } else if (type == "pong") {
    Serial.println("Ping response received");
  }
}

void sendPing() {
  if (!webSocketConnected) return;
  
  StaticJsonDocument<200> doc;
  doc["command"] = "ping";
  
  String jsonString;
  serializeJson(doc, jsonString);
  webSocket.sendTXT(jsonString);
}

void startRecording() {
  if (!webSocketConnected) {
    Serial.println("Cannot start recording - WebSocket not connected");
    return;
  }
  
  // Send start recording command with explicit 16kHz rate
  StaticJsonDocument<300> doc;
  doc["command"] = "start_recording";
  doc["sample_rate"] = 16000;  // Explicitly 16kHz for audio integrity
  doc["chunk_duration_ms"] = CHUNK_DURATION_MS;
  doc["i2s_config"] = "32bit_to_16bit_conversion";
  
  String jsonString;
  serializeJson(doc, jsonString);
  webSocket.sendTXT(jsonString);
  
  Serial.printf("Sent start command: 16kHz, %dms chunks, 32->16 bit conversion\n", CHUNK_DURATION_MS);
  
  isStreamingAudio = true;
  lastChunkTime = millis();
  recordingStartTime = millis();
  totalChunksSent = 0;
  
  Serial.println("Started unlimited audio streaming");
}

void stopRecording() {
  if (!isStreamingAudio) return;
  
  isStreamingAudio = false;
  
  // Calculate recording statistics with detailed analysis
  float recordingDuration = (millis() - recordingStartTime) / 1000.0;
  float expectedDuration = (totalChunksSent * CHUNK_DURATION_MS) / 1000.0;
  float totalDataMB = (totalChunksSent * CHUNK_SIZE / 2) / (1024.0 * 1024.0); // /2 because we convert 32-bit to 16-bit
  
  // Calculate actual vs expected data rates
  int totalSamples16bit = totalChunksSent * (SAMPLE_RATE * CHUNK_DURATION_MS / 1000);
  float actualSampleRate = totalSamples16bit / recordingDuration;
  
  if (webSocketConnected) {
    StaticJsonDocument<300> doc;
    doc["command"] = "stop_recording";
    doc["recording_duration"] = recordingDuration;
    doc["total_chunks"] = totalChunksSent;
    doc["expected_sample_rate"] = SAMPLE_RATE;
    doc["calculated_sample_rate"] = actualSampleRate;
    
    String jsonString;
    serializeJson(doc, jsonString);
    webSocket.sendTXT(jsonString);
  }
  
  Serial.println("=== RECORDING ANALYSIS ===");
  Serial.printf("Duration: %.2fs (expected: %.2fs)\n", recordingDuration, expectedDuration);
  Serial.printf("Chunks: %d, Data: %.2fMB\n", totalChunksSent, totalDataMB);
  Serial.printf("Expected sample rate: %dHz\n", SAMPLE_RATE);
  Serial.printf("Calculated sample rate: %.0fHz\n", actualSampleRate);
  Serial.printf("Sample rate ratio: %.3f (1.000 = perfect)\n", actualSampleRate / SAMPLE_RATE);
  
  if (abs(actualSampleRate - SAMPLE_RATE) > 100) {
    Serial.println("WARNING: Sample rate mismatch detected!");
    Serial.println("This will cause speed/pitch issues in transcription");
  }
}

void streamAudioChunk() {
  if (!isStreamingAudio || !webSocketConnected) return;
  
  // Precise timing - wait for exact chunk duration
  uint32_t now = millis();
  if (now - lastChunkTime < CHUNK_DURATION_MS) return;
  
  size_t bytesRead;
  esp_err_t result = i2s_read(I2S_NUM_0, audioChunk, CHUNK_SIZE, &bytesRead, portMAX_DELAY);
  
  if (result == ESP_OK && bytesRead > 0) {
    // Verify we got the expected amount of data
    int expectedSamples = (SAMPLE_RATE * CHUNK_DURATION_MS) / 1000;  // Expected 16-bit samples
    int32_t* samples = (int32_t*) audioChunk;
    int samplesToSend = bytesRead / 4;  // 4 bytes per 32-bit I2S sample
    
    // Validate timing integrity
    if (samplesToSend != expectedSamples) {
      Serial.printf("WARNING: Sample count mismatch! Expected %d, got %d\n", expectedSamples, samplesToSend);
    }
    
    // Allocate 16-bit PCM buffer with exact sample count
    size_t pcmSize = samplesToSend * 2;  // 2 bytes per 16-bit sample
    int16_t* pcmData = new int16_t[samplesToSend];
    
    // Convert 32-bit I2S samples to 16-bit PCM preserving timing
    // Using configurable bit shift to find the right conversion
    for (int i = 0; i < samplesToSend; i++) {
      // Try different bit shifts - this is the most critical part!
      pcmData[i] = (int16_t)(samples[i] >> BIT_SHIFT_AMOUNT);
    }
    
    // Debug: Log sample values and configuration periodically
    if (totalChunksSent % 20 == 0 && samplesToSend > 0) {
      Serial.printf("Sample debug (shift=%d): Raw I2S: %d (0x%X), Converted: %d\n", 
                    BIT_SHIFT_AMOUNT, samples[0], samples[0], pcmData[0]);
      Serial.printf("Config: %dkHz, MSB format, APLL enabled\n", SAMPLE_RATE);
    }
    
    // Send binary data via WebSocket (maintain chunk order)
    webSocket.sendBIN((uint8_t*)pcmData, pcmSize);
    
    delete[] pcmData;
    lastChunkTime = now;  // Use captured time for precise timing
    totalChunksSent++;
    
    // Timing validation log every 10 chunks
    if (totalChunksSent % 10 == 0) {
      float actualDuration = (now - recordingStartTime) / 1000.0;
      float expectedDuration = (totalChunksSent * CHUNK_DURATION_MS) / 1000.0;
      float timingError = actualDuration - expectedDuration;
      
      Serial.printf("Timing check: %.1fs actual, %.1fs expected, error: %.3fs\n", 
                    actualDuration, expectedDuration, timingError);
      
      if (abs(timingError) > 0.1) {  // More than 100ms drift
        Serial.printf("WARNING: Timing drift detected! This may cause speed issues.\n");
      }
    }
  } else {
    Serial.printf("I2S read error: %d, bytes: %u\n", result, bytesRead);
  }
}

void handleButton() {
  bool raw = digitalRead(BUTTON_PIN);
  
  if (raw != lastRawState) {
    lastDebounceTime = millis();
  }
  
  if (millis() - lastDebounceTime > debounceDelay) {
    if (raw != buttonPressed) {
      buttonPressed = raw;
      
      // Button pressed (LOW due to INPUT_PULLUP)
      if (!buttonPressed && !wasPressed && state == IDLE && webSocketConnected) {
        wasPressed = true;
        startRecording();
        state = RECORDING;
        Serial.println("Button pressed - starting recording");
      }
      // Button released
      else if (buttonPressed && wasPressed) {
        wasPressed = false;
        if (state == RECORDING) {
          stopRecording();
          state = IDLE;
          Serial.println("Button released - stopping recording");
      }
    }
  }
  }
  
  lastRawState = raw;
}

void updateDisplay() {
  display.clearDisplay();
  display.setCursor(0,0);
  display.setTextSize(1);
  
  switch(state) {
    case CONNECTING_WIFI:
      display.println("Connecting WiFi...");
      display.println(ssid);
      break;
      
    case CONNECTING_WEBSOCKET:
      display.println("WiFi Connected");
      display.println("Connecting to server...");
      display.printf("IP: %s:%d\n", serverIP, serverPort);
      break;
      
    case IDLE:
      if (webSocketConnected) {
        display.println("READY - 16kHz");
        display.println("Hold button to");
        display.println("start streaming");
        display.println("");
        display.println("WebSocket: Connected");
      } else {
        display.println("WebSocket Error");
        display.println("Reconnecting...");
      }
      break;

    case RECORDING:
      {
        float duration = (millis() - recordingStartTime) / 1000.0;
        display.println("STREAMING...");
        display.println("Release to stop");
        display.println("");
        display.printf("Time: %.1fs\n", duration);
        display.printf("Chunks: %d\n", totalChunksSent);
        display.printf("Rate: %dkHz\n", SAMPLE_RATE/1000);
      }
      break;
      
    case SHOW_RESULT:
      display.println("Result:");
      display.println("--------");

      // Wrap long text
      if (lastTranscription.length() > 21) {
        display.println(lastTranscription.substring(0, 21));
        if (lastTranscription.length() > 42) {
          display.println(lastTranscription.substring(21, 42));
          if (lastTranscription.length() > 63) {
            display.println(lastTranscription.substring(42, 63));
          }
        } else {
          display.println(lastTranscription.substring(21));
        }
      } else {
        display.println(lastTranscription);
      }
      break;

    case ERROR_STATE:
      display.println("ERROR");
      display.println(statusMessage);
      break;
  }
  
      display.display();
}

void loop() {
  // Handle WebSocket events
  webSocket.loop();
  
  // Handle button input
  handleButton();
  
  // Stream audio if recording
  if (state == RECORDING) {
    streamAudioChunk();
  }
  
  // State machine
  switch(state) {
    case CONNECTING_WIFI:
      if (WiFi.status() == WL_CONNECTED) {
        Serial.println("WiFi connected!");
        Serial.printf("IP address: %s\n", WiFi.localIP().toString().c_str());
        state = CONNECTING_WEBSOCKET;
        setupWebSocket();
      } else if (millis() - stateTimestamp > 30000) {
        // WiFi timeout after 30 seconds
        Serial.println("WiFi connection timeout");
        state = ERROR_STATE;
        statusMessage = "WiFi Timeout";
      }
      break;
      
    case CONNECTING_WEBSOCKET:
      if (webSocketConnected) {
        state = IDLE;
        Serial.println("Ready for voice recording");
      } else if (millis() - stateTimestamp > 15000) {
        // WebSocket timeout after 15 seconds
        Serial.println("WebSocket connection timeout");
        state = ERROR_STATE;
        statusMessage = "Server Timeout";
      }
      break;
      
    case IDLE:
      // Check connection health
      if (!webSocketConnected && millis() - lastReconnectAttempt > RECONNECT_INTERVAL) {
        Serial.println("Attempting WebSocket reconnection...");
        setupWebSocket();
        lastReconnectAttempt = millis();
      }
      break;
      
    case RECORDING:
      // Recording handled in streamAudioChunk()
      break;

    case SHOW_RESULT:
      // Show result for 3 seconds
      if (millis() - stateTimestamp > 3000) {
        state = IDLE;
      }
      break;
      
    case ERROR_STATE:
      // Try to recover after 5 seconds
      if (millis() - stateTimestamp > 5000) {
        if (WiFi.status() != WL_CONNECTED) {
          startWiFiConnection();
        } else {
          state = CONNECTING_WEBSOCKET;
          setupWebSocket();
        }
        stateTimestamp = millis();
      }
      break;
  }
  
  // Update display
  updateDisplay();
  
  // Small delay to prevent watchdog timeout
  delay(10);
}
