#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <driver/i2s.h>

// Wi-Fi
const char* ssid      = "WIFI_SSID";
const char* password  = "WIFI_PASSWORD";
const char* serverUrl = "http://SERVER_IP:PORT/voice_to_text";

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

// Audio buffer
#define SAMPLE_RATE    4000
#define RECORD_SECONDS 8
#define BUFFER_SIZE    (SAMPLE_RATE * RECORD_SECONDS * 2)
uint8_t audioBuffer[BUFFER_SIZE];

// Improved debounce
bool     lastRawState     = HIGH;
uint32_t lastDebounceTime = 0;
const uint32_t debounceDelay = 100;  // Increased to 100ms
bool     buttonPressed    = false;    // Track button press state
uint32_t buttonPressTime  = 0;        // Track when button was pressed

// State machine
enum State { IDLE, RECORDING, SENDING, SHOW_RESULT, CHECKING_SERVER };
State state = CHECKING_SERVER;  // Start by checking server
uint32_t stateTimestamp = 0;     // for timing SHOW_RESULT
size_t   bytesRead      = 0;
String   serverResponse = "";
bool     serverReady    = false;
uint32_t lastServerCheck = 0;
const uint32_t SERVER_CHECK_INTERVAL = 10000;  // Check server every 10 seconds

void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  
  // Add button debugging
  Serial.println("Button pin: " + String(BUTTON_PIN));
  Serial.println("Initial button state: " + String(digitalRead(BUTTON_PIN)));

  // OLED init
  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) while(true);
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);

  // Show startup
  display.setCursor(0,0);
  display.println("Starting...");
  display.display();

  // Wi-Fi
  WiFi.begin(ssid, password);
  while(WiFi.status() != WL_CONNECTED) {
    delay(200);
    Serial.print(".");
  }
  
  Serial.println();
  Serial.println("WiFi connected");
  
  // Show WiFi connected, then check server
  display.clearDisplay();
  display.setCursor(0,0);
  display.println("WiFi connected");
  display.println("Checking server...");
  display.display();

  // I2S init
  i2s_config_t i2s_cfg = {
    .mode                 = (i2s_mode_t)(I2S_MODE_MASTER|I2S_MODE_RX),
    .sample_rate          = SAMPLE_RATE,
    .bits_per_sample      = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count        = 8,
    .dma_buf_len          = 1024,
    .use_apll             = false,
    .tx_desc_auto_clear   = false,
    .fixed_mclk           = 0
  };
  i2s_pin_config_t pin_cfg = {
    .bck_io_num   = I2S_SCK,
    .ws_io_num    = I2S_WS,
    .data_out_num = -1,
    .data_in_num  = I2S_SD
  };
  i2s_driver_install(I2S_NUM_0, &i2s_cfg, 0, nullptr);
  i2s_set_pin(I2S_NUM_0, &pin_cfg);
  i2s_zero_dma_buffer(I2S_NUM_0);
  
  // Initial server check
  checkServer();
}

// Update the WAV header structure with proper byte ordering
struct wav_header_t {
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t chunk_size;  // Total file size - 8
    char wave[4] = {'W', 'A', 'V', 'E'};
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t fmt_chunk_size = 16;
    uint16_t audio_format = 1;  // PCM
    uint16_t num_channels = 1;  // Mono
    uint32_t sample_rate = SAMPLE_RATE;
    uint32_t byte_rate = SAMPLE_RATE * 2;  // SampleRate * NumChannels * BitsPerSample/8
    uint16_t block_align = 2;  // NumChannels * BitsPerSample/8
    uint16_t bits_per_sample = 16;
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t data_chunk_size;  // NumSamples * NumChannels * BitsPerSample/8
};

// Add this function to convert I2S data to proper PCM format
void convert_i2s_to_pcm(const uint8_t* i2s_data, uint8_t* pcm_data, size_t size) {
    for(size_t i = 0; i < size; i += 4) {
        // I2S data is 32-bit, we need 16-bit PCM
        // Swap byte order to test for pitch issue
        pcm_data[i/2] = i2s_data[i+3];
        pcm_data[i/2+1] = i2s_data[i+2];
    }
}

bool checkServer() {
  if (WiFi.status() != WL_CONNECTED) {
    serverReady = false;
    return false;
  }
  
  HTTPClient http;
  String testUrl = String(serverUrl).substring(0, String(serverUrl).lastIndexOf('/')) + "/test";
  http.begin(testUrl);
  http.setTimeout(5000);  // 5 second timeout
  
  int httpCode = http.GET();
  String response = "";
  
  if (httpCode == 200) {
    response = http.getString();
    serverReady = (response == "Server is running!");
    Serial.println("Server check: " + response);
  } else {
    serverReady = false;
    Serial.println("Server check failed: " + String(httpCode));
  }
  
  http.end();
  lastServerCheck = millis();
  return serverReady;
}

void updateDisplay() {
  display.clearDisplay();
  display.setCursor(0,0);
  
  if (WiFi.status() != WL_CONNECTED) {
    display.println("WiFi disconnected");
    display.println("Check connection");
  } else if (!serverReady) {
    display.println("Server not ready");
    display.println("Check server URL:");
    display.setCursor(0,24);
    display.setTextSize(1);
    // Show just the IP part to fit on screen
    String url = String(serverUrl);
    int start = url.indexOf("//") + 2;
    int end = url.indexOf("/", start);
    display.println(url.substring(start, end));
  } else {
    display.println("READY");
    display.println("Press button to");
    display.println("start recording");
  }
  
  display.display();
}

void loop() {
  // Periodic server check (every 10 seconds when idle)
  if (state == IDLE && millis() - lastServerCheck > SERVER_CHECK_INTERVAL) {
    state = CHECKING_SERVER;
  }
  
  // Improved button handling with proper debounce and state protection
  bool raw = digitalRead(BUTTON_PIN);
  
  // Detect button state change
  if (raw != lastRawState) {
    lastDebounceTime = millis();
  }
  
  // Debounce logic
  if (millis() - lastDebounceTime > debounceDelay) {
    static bool lastStable = HIGH;
    
    if (raw != lastStable) {
      lastStable = raw;
      
      // Button pressed (LOW = pressed due to INPUT_PULLUP)
      if (lastStable == LOW && !buttonPressed && state == IDLE && serverReady) {
        buttonPressed = true;
        buttonPressTime = millis();
        Serial.println("Button pressed - starting recording");
        state = RECORDING;
      }
      // Button released
      else if (lastStable == HIGH && buttonPressed) {
        buttonPressed = false;
        Serial.println("Button released");
      }
    }
  }
  lastRawState = raw;

  // State machine
  switch(state) {
    case CHECKING_SERVER:
      display.clearDisplay();
      display.setCursor(0,0);
      display.println("Checking server...");
      display.display();
      
      checkServer();
      state = IDLE;
      break;
      
    case IDLE:
      updateDisplay();
      break;

    case RECORDING:
      display.clearDisplay();
      display.setCursor(0,0);
      display.println("Recording...");
      display.display();

      // Perform blocking read (2 seconds)
      i2s_read(I2S_NUM_0, audioBuffer, BUFFER_SIZE, &bytesRead, portMAX_DELAY);
      Serial.printf("Read %u bytes\n", bytesRead);
      
      // Reset button state after recording starts
      buttonPressed = false;

      state = SENDING;
      break;

    case SENDING: {
      display.clearDisplay();
      display.setCursor(0,0);
      display.println("Sending...");
      display.display();

      if (WiFi.status() == WL_CONNECTED) {
          // Convert I2S data to PCM
          size_t pcm_size = bytesRead / 2;  // 32-bit to 16-bit
          uint8_t* pcm_buffer = new uint8_t[pcm_size];
          convert_i2s_to_pcm(audioBuffer, pcm_buffer, bytesRead);

          // Create WAV header
          wav_header_t wav_header;
          wav_header.data_chunk_size = pcm_size;
          wav_header.chunk_size = pcm_size + 36;  // data size + header size - 8

          // Create final buffer with header + PCM data
          size_t total_size = sizeof(wav_header) + pcm_size;
          uint8_t* wav_buffer = new uint8_t[total_size];
          memcpy(wav_buffer, &wav_header, sizeof(wav_header));
          memcpy(wav_buffer + sizeof(wav_header), pcm_buffer, pcm_size);

          // Send to server
          HTTPClient http;
          http.begin(serverUrl);
          http.addHeader("Content-Type", "audio/wav");
          http.addHeader("Content-Length", String(total_size));
          int code = http.POST(wav_buffer, total_size);
          
          // Clean up
          delete[] pcm_buffer;
          delete[] wav_buffer;

          if (code == 200) {
              serverResponse = http.getString();
          } else {
              serverResponse = "HTTP err:" + String(code);
              // Mark server as not ready if we get an error
              serverReady = false;
          }
          http.end();
      } else {
          serverResponse = "WiFi lost";
          serverReady = false;
      }
      Serial.println("Resp: " + serverResponse);

      stateTimestamp = millis();
      state = SHOW_RESULT;
      break;
    }

    case SHOW_RESULT:
      display.clearDisplay();
      display.setCursor(0,0);
      display.println("Server:");
      display.setCursor(0,16);
      display.println(serverResponse);
      display.display();

      // after 3 seconds go back to IDLE
      if (millis() - stateTimestamp > 3000) {
        state = IDLE;
      }
      break;
  }

  // Safety check: if we're stuck in recording for too long, reset
  if (state == RECORDING && millis() - buttonPressTime > 10000) {
    Serial.println("Safety timeout - resetting to IDLE");
    state = IDLE;
    buttonPressed = false;
  }
  
  // tiny idle delay so we don't starve I2S or WDT
  delay(10);
}
