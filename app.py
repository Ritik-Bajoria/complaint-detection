from flask import Flask, jsonify, request
import os
import signal
import sys
from flask_cors import CORS
from utils.logger import Logger
from utils.preprocessdata import *
from utils.classification import classify
from flask_swagger_ui import get_swaggerui_blueprint

# initialize flask app
app = Flask(__name__)
CORS(app)

# get the api key
API_KEY = os.getenv('API_KEY')
print(API_KEY)
# create logger instance
logger=Logger()

# for swagger documentation
SWAGGER_URL = '/api/docs'  # URL for exposing Swagger UI (without trailing '/')
API_URL = '/static/swagger.json'  # Our API url (can of course be a local resource)

# Middleware: before each request
@app.before_request
def before_request_func():
    # This will run before every request
    logger.info(f"Request from {request.remote_addr} at {request.method} {request.url}")

# Error response in case of route not found
@app.errorhandler(404)
def not_found_error(e):
    return jsonify({
        "error": True,
        "message": "URL not found"
    }), 404

# Error response in case of method not allowed
@app.errorhandler(405)
def method_not_allowed_error(e):
    return jsonify({
        "error": True,
        "message": "Method not allowed"
    }), 405

def validate_api_key():
    # Validate the API key from the request headers.
    api_key = request.headers.get('X-API-KEY')
    if api_key is None or api_key != API_KEY:
        return False
    return True

@app.route("/api/complaint", methods=['POST'])
def complaint_detector():
    # Validate the API key
    if not validate_api_key():
        return jsonify({
            "error":True,
            "message": "Unauthorized access"
        }), 401
    try:
        # Get the text to be checked from the request
        if 'text' not in request.form:
            return jsonify({
                "error": True,
                'message': 'No text in the request'
                }), 400
        
        # load text to a variable
        text = request.form['text']

        # preprocess text
        text = clean_text(text)
        text = remove_stopwords(text)
        text = apply_stemmer(text)
        text = correct_spell(text)
        text = apply_lemmatizer(text)

        # classify as complaint or non-complaint
        classification = classify(text)

        return jsonify({
            "classification":classification
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Call factory function to create our blueprint
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,  # Swagger UI static files will be mapped to '{SWAGGER_URL}/dist/'
    API_URL,
    config={  # Swagger UI config overrides
        'app_name': "Test application"
    },
    # oauth_config={  # OAuth config. See https://github.com/swagger-api/swagger-ui#oauth2-configuration .
    #    'clientId': "your-client-id",
    #    'clientSecret': "your-client-secret-if-required",
    #    'realm': "your-realms",
    #    'appName': "your-app-name",
    #    'scopeSeparator': " ",
    #    'additionalQueryStringParams': {'test': "hello"}
    # }
)

app.register_blueprint(swaggerui_blueprint)

# Graceful shutdown function
def graceful_shutdown(signal, frame):
    logger.info("Shutting down gracefully...")
    # Perform any cleanup here if needed
    sys.exit(0)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)

if __name__ == '__main__':
    host = os.getenv('HOST')
    port = os.getenv('PORT')
    app.run(debug=True,host=host,port=port)