# AI Recruiter Audio Analysis Model

An advanced audio analysis system for evaluating candidate communication skills during recruitment interviews. This model analyzes speech fluency, grammar, and professionalism to provide comprehensive feedback in JSON format.

## Features

- **Audio-Only Fluency Analysis**: Advanced audio-based fluency assessment using speech patterns, rhythm, and energy analysis
- **Speech Transcription**: Uses OpenAI Whisper for accurate audio-to-text conversion (internal use only)
- **Grammar Analysis**: Checks for grammatical errors and sentence structure using LanguageTool
- **Professionalism Analysis**: Assesses language formality, confidence indicators, and vocabulary diversity
- **JSON Output**: Returns structured JSON data for easy integration
- **Flask API**: RESTful API endpoints for easy integration
- **Comprehensive Scoring**: Weighted scoring system across all dimensions

## Output Structure

The model generates a JSON report with the following structure:

```json
{
  "overall_score": 83.5,
  "report": {
    "fluency_analysis": {
      "score": 85,
      "analysis": [
        "Excellent speech activity - good use of speaking time",
        "Good speech rate - clear and understandable",
        "Excellent rhythm consistency - smooth flow",
        "Good pause control - reasonable pauses",
        "Natural energy variation - engaging delivery"
      ]
    },
    "grammar_analysis": {
      "score": 94,
      "analysis": [
        "Good grammar with minor errors",
        "Good sentence structure and variety"
      ],
      "errors": [
        "Possible spelling mistake found."
      ],
      "error_count": 1
    },
    "professionalism_analysis": {
      "score": 80,
      "analysis": [
        "Excellent professional language - no filler words",
        "Confident and assertive communication style",
        "Excellent vocabulary diversity"
      ]
    }
  }
}
```

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. For Windows users, you may need to install additional dependencies for audio processing:

```bash
pip install pydub
```

## Usage

### Command Line Usage

```bash
python audio_analysis.py path/to/audio/file.wav
```

### Python Script Usage

```python
from audio_analysis import analyze_audio, analyze_audio_with_text

# Analyze an audio file
result = analyze_audio("candidate_audio.wav")
print(f"Overall Score: {result['overall_score']}/100")

# Analyze with custom text
result = analyze_audio_with_text("candidate_audio.wav", "Custom transcribed text")
print(f"Overall Score: {result['overall_score']}/100")
```

### Flask API Usage

1. Start the API server:

```bash
python app.py
```

2. The API will be available at `http://localhost:5000`

#### API Endpoints

- **GET /** - API documentation
- **GET /health** - Health check
- **POST /analyze** - Analyze audio file (multipart/form-data)
- **POST /analyze-text** - Analyze text only (JSON)

#### Example API Usage

```bash
# Analyze audio file
curl -X POST -F "audio=@candidate_audio.wav" http://localhost:5000/analyze

# Analyze with custom text
curl -X POST -F "audio=@candidate_audio.wav" -F "text=Custom transcribed text" http://localhost:5000/analyze

# Analyze text only
curl -X POST -H "Content-Type: application/json" \
  -d '{"text":"Hello, my name is John Smith and I am excited to discuss my qualifications."}' \
  http://localhost:5000/analyze-text
```

### Test the API

```bash
python test_api.py
```

## Supported Audio Formats

- WAV
- MP3
- M4A
- FLAC
- OGG

## Analysis Components

### 1. Fluency Analysis (50% weight for overall score)
- **Speech Activity Ratio**: Measures how much of the audio contains actual speech vs silence
- **Speech Rate**: Natural speaking pace based on audio patterns (optimal: 65-90%)
- **Rhythm Consistency**: How consistent the speech bursts are (measures natural flow)
- **Pause Control**: Frequency and duration of pauses (distinguishes natural vs nervous pauses)
- **Energy Variation**: Natural variation in voice energy (indicates engaging, expressive speech)

### 2. Grammar Analysis (30% weight for overall score)
- **Error Detection**: Uses LanguageTool for comprehensive grammar checking
- **Sentence Structure**: Analyzes sentence length and variety
- **Error Patterns**: Identifies common grammatical issues

### 3. Professionalism Analysis (20% weight for overall score)
- **Filler Words**: Detects informal language like "um", "uh", "like", "you know"
- **Confidence Indicators**: Evaluates hedging language vs. assertive communication
- **Vocabulary Sophistication**: Measures language diversity and formality

## Scoring System

### Overall Score Ranges
- **Excellent**: 85-100 points
- **Good**: 70-84 points
- **Moderate**: 55-69 points
- **Below Average**: 40-54 points
- **Poor**: 20-39 points

### Component Scoring
- **Overall Score**: Based on Fluency (50%) + Grammar (30%) + Professionalism (20%)
- **Individual Component Scores**: 0-100 points each

## Audio-Only Fluency Analysis

The fluency analysis is based purely on audio characteristics without requiring transcription:

### What It Analyzes:
- **Speech Activity Ratio**: Percentage of time spent speaking vs silence
- **Speech Rate**: Natural speaking pace based on audio patterns
- **Rhythm Consistency**: How smoothly speech flows between segments
- **Pause Control**: Frequency and duration of pauses
- **Energy Variation**: Natural variation in voice energy

### Key Features:
- ✅ No transcription needed for fluency analysis
- ✅ Real audio patterns - speech bursts, energy levels, rhythm
- ✅ Natural speech detection - distinguishes speech from silence
- ✅ Rhythm analysis - measures consistency of speech flow
- ✅ Pause intelligence - distinguishes natural pauses from nervous hesitations
- ✅ Energy patterns - detects natural variation vs monotone speech

## Performance

- **Processing Time**: Typically 10-30 seconds for 1-3 minute audio files
- **Accuracy**: High accuracy for clear speech, moderate for noisy environments
- **GPU Support**: Automatically uses CUDA if available for faster processing
- **Memory Usage**: Efficient processing with minimal memory footprint

## Example Output

```json
{
  "overall_score": 83.5,
  "report": {
    "fluency_analysis": {
      "score": 85,
      "analysis": [
        "Excellent speech activity - good use of speaking time",
        "Good speech rate - clear and understandable",
        "Excellent rhythm consistency - smooth flow",
        "Good pause control - reasonable pauses",
        "Natural energy variation - engaging delivery"
      ]
    },
    "grammar_analysis": {
      "score": 94,
      "analysis": [
        "Good grammar with minor errors",
        "Good sentence structure and variety"
      ],
      "errors": [
        "Possible spelling mistake found."
      ],
      "error_count": 1
    },
    "professionalism_analysis": {
      "score": 80,
      "analysis": [
        "Excellent professional language - no filler words",
        "Confident and assertive communication style",
        "Excellent vocabulary diversity"
      ]
    }
  }
}
```

## Troubleshooting

1. **Audio Quality**: Ensure clear audio with minimal background noise
2. **File Format**: Use supported formats (WAV, MP3, M4A, FLAC, OGG)
3. **Dependencies**: Install all required packages from requirements.txt
4. **GPU Issues**: The model works on CPU but is faster with GPU
5. **API Issues**: Check that the Flask server is running and accessible

## API Testing with Postman

1. **Set up the request**:
   - Method: POST
   - URL: `http://localhost:5000/analyze`
   - Body: form-data

2. **Add the audio file**:
   - Key: `audio`
   - Type: File (important!)
   - Value: Select your audio file

3. **Optional: Add custom text**:
   - Key: `text`
   - Type: Text
   - Value: Your custom transcription

## Future Enhancements

- Emotion detection
- Accent analysis
- Language proficiency assessment
- Real-time analysis capabilities
- Integration with video analysis
- Multi-language support
