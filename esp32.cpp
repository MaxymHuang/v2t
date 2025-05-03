#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <driver/i2s.h>

// Wi-Fi
const char* ssid      = "gay wifi";
const char* password  = "31415926";
const char* serverUrl = "http://10.0.10.98:5000/voice_to_text";

// OLED
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// I2S Mic
#define I2S_WS   25
#define I2S_SD   33
#define I2S_SCK  26

// Button
#define BUTTON_PIN 0

// Audio buffer
#define SAMPLE_RATE    16000
#define RECORD_SECONDS 2
#define BUFFER_SIZE    (SAMPLE_RATE * RECORD_SECONDS * 2)
uint8_t audioBuffer[BUFFER_SIZE];

// Debounce
bool     lastRawState     = HIGH;
uint32_t lastDebounceTime = 0;
const uint32_t debounceDelay = 50;

// State machine
enum State { IDLE, RECORDING, SENDING, SHOW_RESULT } ;
State state = IDLE;
uint32_t stateTimestamp = 0;     // for timing SHOW_RESULT
size_t   bytesRead      = 0;
String   serverResponse = "";

void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT_PULLUP);

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
  display.clearDisplay();
  display.setCursor(0,0);
  display.println("WiFi connected");
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
        // Skip first 2 bytes of each I2S word (padding)
        // and take only the valid 16-bit sample
        pcm_data[i/2] = i2s_data[i+2];
        pcm_data[i/2+1] = i2s_data[i+3];
    }
}

void loop() {
  // 1) Read raw button & debounce
  bool raw = digitalRead(BUTTON_PIN);
  if (raw != lastRawState) {
    lastDebounceTime = millis();
  }
  if (millis() - lastDebounceTime > debounceDelay) {
    static bool lastStable = HIGH;
    if (raw != lastStable) {
      lastStable = raw;
      if (lastStable == LOW && state == IDLE) {
        // button pressed → begin recording
        state = RECORDING;
        Serial.println("→ RECORDING");
      }
    }
  }
  lastRawState = raw;

  // State machine
  switch(state) {
    case IDLE:
      // Show prompt if desired
      break;

    case RECORDING:
      display.clearDisplay();
      display.setCursor(0,0);
      display.println("Recording...");
      display.display();

      // Perform blocking read (2 seconds)
      i2s_read(I2S_NUM_0, audioBuffer, BUFFER_SIZE, &bytesRead, portMAX_DELAY);
      Serial.printf("Read %u bytes\n", bytesRead);

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
          }
          http.end();
      } else {
          serverResponse = "WiFi lost";
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

  // tiny idle delay so we don't starve I2S or WDT
  delay(10);
}
