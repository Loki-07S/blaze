from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import tempfile
from audio_analysis import analyze_audio, analyze_audio_with_text

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg'}

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Audio Analysis API is running"
    })

@app.route('/analyze', methods=['POST'])
def analyze_audio_endpoint():
    """
    Analyze audio file endpoint
    
    Expected request:
    - audio file in multipart/form-data
    - Optional: text field for custom transcription
    
    Returns:
    - JSON with analysis results
    """
    try:
        # Check if audio file is present
        if 'audio' not in request.files:
            return jsonify({
                "error": "No audio file provided",
                "message": "Please upload an audio file using the 'audio' field"
            }), 400
        
        audio_file = request.files['audio']
        
        # Check if file is selected
        if audio_file.filename == '':
            return jsonify({
                "error": "No file selected",
                "message": "Please select an audio file to upload"
            }), 400
        
        # Check file extension
        if not allowed_file(audio_file.filename):
            return jsonify({
                "error": "Invalid file type",
                "message": f"Allowed file types: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
        
        # Save the uploaded file temporarily
        filename = secure_filename(audio_file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(temp_path)
        
        try:
            # Check if custom text is provided
            custom_text = request.form.get('text', None)
            
            if custom_text:
                # Use provided text for grammar and professionalism analysis
                result = analyze_audio_with_text(temp_path, custom_text)
            else:
                # Use automatic transcription
                result = analyze_audio(temp_path)
            
            return jsonify(result)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        return jsonify({
            "error": "Analysis failed",
            "message": str(e)
        }), 500

@app.route('/analyze-text', methods=['POST'])
def analyze_text_only():
    """
    Analyze text only (without audio) for grammar and professionalism
    
    Expected request:
    - JSON with 'text' field
    
    Returns:
    - JSON with grammar and professionalism analysis only
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "error": "No text provided",
                "message": "Please provide text in JSON format with 'text' field"
            }), 400
        
        text = data['text']
        
        if not text.strip():
            return jsonify({
                "error": "Empty text",
                "message": "Please provide non-empty text"
            }), 400
        
        # Import the analysis functions
        from audio_analysis import analyze_grammar_advanced, analyze_professionalism, calculate_overall_score
        
        # Analyze grammar and professionalism
        grammar_analysis = analyze_grammar_advanced(text)
        professionalism_analysis = analyze_professionalism(text)
        
        # Calculate overall score (fluency score will be 0 since no audio)
        overall_score = calculate_overall_score(0, grammar_analysis["score"], professionalism_analysis["score"])
        
        result = {
            "score": overall_score,
            "report": {
                "grammar_analysis": {
                    "score": grammar_analysis["score"],
                    "analysis": grammar_analysis["analysis"],
                    "errors": grammar_analysis["errors"],
                    "error_count": grammar_analysis["error_count"]
                },
                "professionalism_analysis": {
                    "score": professionalism_analysis["score"],
                    "analysis": professionalism_analysis["analysis"]
                }
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "error": "Analysis failed",
            "message": str(e)
        }), 500

@app.route('/', methods=['GET'])
def index():
    """API documentation endpoint"""
    return jsonify({
        "message": "Audio Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "API documentation (this endpoint)",
            "GET /health": "Health check",
            "POST /analyze": "Analyze audio file (with optional custom text)",
            "POST /analyze-text": "Analyze text only (grammar and professionalism)"
        },
        "usage": {
            "/analyze": "Upload audio file using multipart/form-data with 'audio' field. Optionally include 'text' field for custom transcription.",
            "/analyze-text": "Send JSON with 'text' field for grammar and professionalism analysis only."
        },
        "supported_audio_formats": list(ALLOWED_EXTENSIONS)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
