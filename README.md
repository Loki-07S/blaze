# AI Recruiter Audio Analysis Model

An advanced audio analysis system for evaluating candidate communication skills during recruitment interviews. This model analyzes speech fluency, grammar, professionalism, non-verbal cues, and vocabulary to provide comprehensive feedback.

## Features

- **Speech Transcription**: Uses OpenAI Whisper for accurate audio-to-text conversion
- **Fluency Analysis**: Evaluates speaking pace, pauses, and speech flow
- **Grammar Analysis**: Checks for grammatical errors and sentence structure
- **Professionalism Analysis**: Assesses language formality and confidence indicators
- **Non-verbal Cues Analysis**: Evaluates confidence and enthusiasm through speech patterns
- **Vocabulary Analysis**: Measures vocabulary diversity and sophistication
- **Comprehensive Scoring**: Weighted scoring system across all dimensions

## Output Structure

The model generates a comprehensive text report with the following structure:

```
=== Audio Analysis Report ===

Overall Score: 85.5/100

Transcribed Text:
[Transcribed audio content]

=== Detailed Analysis ===

Fluency Analysis:
Score: 90/100
- Excellent speaking pace - natural and engaging
- Good flow with reasonable pauses
WPM: 150.2
Pause Frequency: 3.1 per minute

Grammar Analysis:
Score: 96/100
- Excellent grammar - no errors detected
- Good sentence structure and variety
Errors: 0

Professionalism Analysis:
Score: 85/100
- Excellent professional language - no filler words
- Confident and assertive communication style
Informal Words: 0

Non-Verbal Cues Analysis:
Score: 80/100
- Good confidence indicators - natural flow
- Minimal hesitation - good preparation and confidence
Pause Frequency: 3.1 per minute

Vocabulary Analysis:
Score: 75/100
- Good vocabulary diversity
- Good use of advanced vocabulary
Vocabulary Richness: 0.65
Advanced Words: 2
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
python audio_analysis_simple.py path/to/audio/file.wav --report output_report.txt
```

### Python Script Usage

```python
from audio_analysis_simple import main

# Analyze an audio file
report = main("candidate_audio.wav", "candidate_report.txt")
print(f"Overall Score: {report['score']}/100")
```

### Test the Model

```bash
python test_audio_analysis.py
```

## Supported Audio Formats

- WAV
- MP3
- M4A
- FLAC

## Analysis Components

### 1. Fluency Analysis (50% weight for overall score)
- **Speaking Pace**: Optimal range 100-200 words per minute
- **Pause Analysis**: Evaluates natural flow vs. nervous hesitation
- **Speech Flow**: Measures continuity and smoothness

### 2. Grammar Analysis (30% weight for overall score)
- **Error Detection**: Uses LanguageTool for comprehensive grammar checking
- **Sentence Structure**: Analyzes sentence length and variety
- **Error Patterns**: Identifies common grammatical issues

### 3. Professionalism Analysis (20% weight for overall score)
- **Filler Words**: Detects informal language like "um", "uh", "like"
- **Confidence Indicators**: Evaluates hedging language vs. assertive communication
- **Vocabulary Sophistication**: Measures language diversity and formality

### 4. Non-verbal Cues Analysis (Report only)
- **Confidence Indicators**: Based on speech pace and pause patterns
- **Hesitation Analysis**: Evaluates preparation and nervousness
- **Enthusiasm**: Measures expressive language and engagement

### 5. Vocabulary Analysis (Report only)
- **Vocabulary Diversity**: Measures unique word usage
- **Advanced Vocabulary**: Detects sophisticated word choices
- **Repetition Analysis**: Identifies overused words

## Scoring System

- **Overall Score**: Based on Fluency (50%) + Grammar (30%) + Professionalism (20%) (0-100)
- **Individual Component Scores**: 
  - Fluency: 0-100 points (50% weight for overall score)
  - Grammar: 0-100 points (30% weight for overall score)
  - Professionalism: 0-100 points (20% weight for overall score)
  - Non-verbal Cues: 0-100 points (Report only)
  - Vocabulary: 0-100 points (Report only)

## Free Alternative to OpenAI

This improved version uses free alternatives instead of OpenAI:

- **Whisper**: For speech transcription (free, runs locally)
- **LanguageTool**: For grammar checking (free)
- **TextBlob**: For text analysis (free)
- **Parselmouth**: For audio analysis (free)
- **Custom Algorithms**: For professionalism, non-verbal cues, and vocabulary analysis

## Performance

- **Processing Time**: Typically 30-60 seconds for 2-3 minute audio files
- **Accuracy**: High accuracy for clear speech, moderate for noisy environments
- **GPU Support**: Automatically uses CUDA if available for faster processing

## Example Output

```
=== Audio Analysis Report ===

Overall Score: 87.3/100

Transcribed Text:
"Hello, my name is John Smith and I'm excited to discuss my qualifications for this position..."

=== Detailed Analysis ===

Fluency Analysis:
Score: 90/100
- Excellent speaking pace - natural and engaging
- Good flow with reasonable pauses
WPM: 145.2
Pause Frequency: 2.8 per minute

Grammar Analysis:
Score: 96/100
- Excellent grammar - no errors detected
- Good sentence structure and variety
Errors: 0

Professionalism Analysis:
Score: 85/100
- Excellent professional language - no filler words
- Confident and assertive communication style
Informal Words: 0

Non-Verbal Cues Analysis:
Score: 80/100
- Good confidence indicators - natural flow
- Minimal hesitation - good preparation and confidence
Pause Frequency: 2.8 per minute

Vocabulary Analysis:
Score: 75/100
- Good vocabulary diversity
- Good use of advanced vocabulary
Vocabulary Richness: 0.65
Advanced Words: 2
```

## Troubleshooting

1. **Audio Quality**: Ensure clear audio with minimal background noise
2. **File Format**: Use supported formats (WAV, MP3, M4A, FLAC)
3. **Dependencies**: Install all required packages from requirements.txt
4. **GPU Issues**: The model works on CPU but is faster with GPU

## Future Enhancements

- Emotion detection
- Accent analysis
- Language proficiency assessment
- Real-time analysis capabilities
- Integration with video analysis
