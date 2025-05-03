from flask import Flask, request, jsonify
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

@app.route('/test', methods=['GET'])
def test_get():
    return "Server is running!"

@app.route('/voice_to_text', methods=['POST'])
def voice_to_text():
    try:
        # Log request details
        logger.debug(f"Received request: Content-Type: {request.content_type}")
        logger.debug(f"Request length: {len(request.data)} bytes")

        # Save received audio for inspection
        audio_data = request.data
        with open("received_data.bin", "wb") as f:
            f.write(audio_data)

        # Log the first few bytes for debugging
        if len(audio_data) > 0:
            logger.debug(f"First 20 bytes: {audio_data[:20].hex()}")
            return jsonify({
                "status": "success",
                "bytes_received": len(audio_data),
                "first_bytes": audio_data[:20].hex()
            })
        else:
            logger.warning("Received empty data")
            return jsonify({"status": "error", "message": "Empty data received"})

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting test server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
