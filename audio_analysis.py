import whisper
import numpy as np
import language_tool_python
from pydub import AudioSegment
import os
import time
import json
import torch
from textblob import TextBlob
import re

if torch.cuda.is_available():
    print("CUDA is available. GPU will be used for inference.")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is NOT available. Running on CPU.")

def transcribe_audio(audio_path):
    model = whisper.load_model("tiny")
    result = model.transcribe(audio_path)
    return result['text']

def analyze_fluency(audio_path):
    """Simple fluency analysis using pydub and transcript"""
    # Use pydub to get audio duration
    audio = AudioSegment.from_file(audio_path)
    duration = len(audio) / 1000.0  # Convert to seconds
    
    # Get transcript for analysis
    transcript = transcribe_audio(audio_path)
    words = transcript.split()
    num_words = len(words)
    
    # Estimate pauses based on sentence structure
    sentences = re.split(r'[.!?]+', transcript)
    sentences = [s.strip() for s in sentences if s.strip()]
    estimated_pauses = len(sentences) - 1  # Pauses between sentences
    
    words_per_minute = (num_words / duration) * 60 if duration > 0 else 0
    
    return {
        "num_pauses": estimated_pauses,
        "duration_sec": duration,
        "words_per_minute": words_per_minute
    }

def analyze_fluency_advanced(transcript, fluency_stats):
    """Advanced fluency analysis using text and audio metrics"""
    wpm = fluency_stats['words_per_minute']
    pauses = fluency_stats['num_pauses']
    duration = fluency_stats['duration_sec']
    
    # Analyze speech patterns
    sentences = re.split(r'[.!?]+', transcript)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0
    
    # Fluency scoring - More lenient ranges (out of 100)
    fluency_score = 0
    fluency_analysis = []
    
    # WPM analysis - More realistic ranges
    if 100 <= wpm <= 200:  # Expanded from 120-180
        fluency_score += 50
        fluency_analysis.append("Excellent speaking pace - natural and engaging")
    elif 80 <= wpm < 100 or 200 < wpm <= 250:  # Expanded ranges
        fluency_score += 40
        fluency_analysis.append("Good speaking pace - clear and understandable")
    elif 60 <= wpm < 80 or 250 < wpm <= 300:  # More lenient
        fluency_score += 30
        fluency_analysis.append("Moderate speaking pace - could be improved")
    else:
        fluency_score += 20
        fluency_analysis.append("Speaking pace needs improvement - too slow or too fast")
    
    # Pause analysis - More lenient
    pause_frequency = pauses / (duration / 60) if duration > 0 else 0  # pauses per minute
    if pause_frequency <= 4:  # Increased from 2
        fluency_score += 50
        fluency_analysis.append("Excellent flow with minimal pauses")
    elif pause_frequency <= 8:  # Increased from 4
        fluency_score += 40
        fluency_analysis.append("Good flow with reasonable pauses")
    elif pause_frequency <= 12:  # Increased from 6
        fluency_score += 30
        fluency_analysis.append("Moderate flow with some hesitation")
    else:
        fluency_score += 20
        fluency_analysis.append("Frequent pauses indicate nervousness or lack of preparation")
    
    return {
        "score": fluency_score,
        "analysis": fluency_analysis,
        "metrics": {
            "words_per_minute": wpm,
            "pause_frequency": pause_frequency,
            "avg_sentence_length": avg_sentence_length
        }
    }

def analyze_grammar_advanced(text):
    """Advanced grammar analysis using multiple tools"""
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    
    # TextBlob for additional analysis
    blob = TextBlob(text)
    
    grammar_score = 100  # Start with 100 instead of 50
    grammar_analysis = []
    
    # Error count analysis - More lenient
    num_errors = len(matches)
    if num_errors == 0:
        grammar_analysis.append("Excellent grammar - no errors detected")
    elif num_errors <= 3:  # Increased from 2
        grammar_score -= 6  # Reduced penalty (doubled from 3)
        grammar_analysis.append("Good grammar with minor errors")
    elif num_errors <= 6:  # Increased from 5
        grammar_score -= 12  # Reduced penalty (doubled from 6)
        grammar_analysis.append("Moderate grammar issues")
    elif num_errors <= 12:  # Increased from 10
        grammar_score -= 24  # Reduced penalty (doubled from 12)
        grammar_analysis.append("Significant grammar problems")
    else:
        grammar_score -= 40  # Reduced penalty (doubled from 20)
        grammar_analysis.append("Major grammar issues need attention")
    
    # Sentence structure analysis
    sentences = blob.sentences
    avg_sentence_length = np.mean([len(s.words) for s in sentences]) if sentences else 0
    
    if 8 <= avg_sentence_length <= 25:  # Expanded from 10-20
        grammar_analysis.append("Good sentence structure and variety")
    elif avg_sentence_length < 8:
        grammar_analysis.append("Sentences are too short - consider combining ideas")
    else:
        grammar_analysis.append("Sentences are quite long - consider breaking them up")
    
    # Specific error types
    error_types = {}
    for match in matches:
        error_type = match.ruleId
        if error_type not in error_types:
            error_types[error_type] = 0
        error_types[error_type] += 1
    
    if error_types:
        most_common_error = max(error_types, key=error_types.get)
        grammar_analysis.append(f"Most common error: {most_common_error}")
    
    return {
        "score": grammar_score,
        "analysis": grammar_analysis,
        "errors": [match.message for match in matches],
        "error_count": num_errors
    }

def analyze_professionalism(text):
    """Analyze professionalism based on language use"""
    professionalism_score = 50
    professionalism_analysis = []
    
    # Check for informal language - More lenient
    informal_words = ['um', 'uh', 'like', 'you know', 'basically', 'actually', 'literally']
    informal_count = sum(text.lower().count(word) for word in informal_words)
    
    if informal_count == 0:
        professionalism_score += 10
        professionalism_analysis.append("Excellent professional language - no filler words")
    elif informal_count <= 5:  # Increased from 3
        professionalism_score += 5
        professionalism_analysis.append("Good professional language with minimal filler words")
    elif informal_count <= 10:  # Increased from 6
        professionalism_analysis.append("Moderate use of filler words - could be more professional")
    else:
        professionalism_score -= 5  # Reduced penalty
        professionalism_analysis.append("Excessive use of filler words - needs improvement")
    
    # Check for confident language - More lenient
    confident_phrases = ['i believe', 'i think', 'i feel', 'maybe', 'perhaps', 'might']
    confident_count = sum(text.lower().count(phrase) for phrase in confident_phrases)
    
    if confident_count <= 4:  # Increased from 2
        professionalism_score += 10
        professionalism_analysis.append("Confident and assertive communication style")
    elif confident_count <= 8:  # Increased from 5
        professionalism_score += 5
        professionalism_analysis.append("Generally confident with some hedging")
    else:
        professionalism_score -= 3  # Reduced penalty
        professionalism_analysis.append("Overuse of hedging language - be more confident")
    
    # Check vocabulary sophistication
    words = text.lower().split()
    unique_words = len(set(words))
    total_words = len(words)
    vocabulary_richness = unique_words / total_words if total_words > 0 else 0
    
    if vocabulary_richness >= 0.6:  # Reduced from 0.7
        professionalism_score += 10
        professionalism_analysis.append("Excellent vocabulary diversity")
    elif vocabulary_richness >= 0.4:  # Reduced from 0.5
        professionalism_score += 5
        professionalism_analysis.append("Good vocabulary diversity")
    else:
        professionalism_score -= 3  # Reduced penalty
        professionalism_analysis.append("Limited vocabulary - consider expanding word choice")
    
    return {
        "score": min(100, max(0, professionalism_score)),
        "analysis": professionalism_analysis,
        "metrics": {
            "informal_words": informal_count,
            "confident_phrases": confident_count,
            "vocabulary_richness": vocabulary_richness
        }
    }

def analyze_non_verbal_cues(fluency_stats, transcript):
    """Analyze non-verbal cues based on audio metrics and speech patterns"""
    non_verbal_score = 50
    non_verbal_analysis = []
    
    wpm = fluency_stats['words_per_minute']
    pauses = fluency_stats['num_pauses']
    duration = fluency_stats['duration_sec']
    
    # Confidence indicators - More lenient
    if 100 <= wpm <= 200 and pauses <= 6:  # Expanded ranges
        non_verbal_score += 20
        non_verbal_analysis.append("High confidence indicators - steady pace and minimal pauses")
    elif 80 <= wpm <= 250 and pauses <= 12:  # Expanded ranges
        non_verbal_score += 10
        non_verbal_analysis.append("Good confidence indicators - natural flow")
    elif wpm < 80:
        non_verbal_score -= 5  # Reduced penalty
        non_verbal_analysis.append("Slow speech may indicate nervousness or lack of confidence")
    elif wpm > 250:
        non_verbal_score -= 3  # Reduced penalty
        non_verbal_analysis.append("Very fast speech may indicate nervousness")
    
    # Hesitation analysis - More lenient
    pause_frequency = pauses / (duration / 60) if duration > 0 else 0
    if pause_frequency <= 4:  # Increased from 2
        non_verbal_analysis.append("Minimal hesitation - good preparation and confidence")
    elif pause_frequency <= 8:  # Increased from 4
        non_verbal_analysis.append("Moderate hesitation - generally confident")
    elif pause_frequency <= 12:  # Increased from 6
        non_verbal_analysis.append("Frequent hesitation - may need more preparation")
    else:
        non_verbal_score -= 10  # Reduced penalty
        non_verbal_analysis.append("Excessive hesitation - indicates nervousness or unpreparedness")
    
    # Enthusiasm indicators
    exclamation_count = transcript.count('!')
    if exclamation_count >= 2:
        non_verbal_analysis.append("Shows enthusiasm through expressive language")
    elif exclamation_count == 1:
        non_verbal_analysis.append("Moderate enthusiasm shown")
    else:
        non_verbal_analysis.append("Could show more enthusiasm in delivery")
    
    return {
        "score": min(100, max(0, non_verbal_score)),
        "analysis": non_verbal_analysis,
        "metrics": {
            "pause_frequency": pause_frequency,
            "exclamation_count": exclamation_count
        }
    }

def analyze_vocabulary(text):
    """Analyze vocabulary sophistication and usage"""
    vocabulary_score = 50
    vocabulary_analysis = []
    
    # Word count and diversity
    words = text.lower().split()
    unique_words = len(set(words))
    total_words = len(words)
    vocabulary_richness = unique_words / total_words if total_words > 0 else 0
    
    if vocabulary_richness >= 0.7:  # Reduced from 0.8
        vocabulary_score += 20
        vocabulary_analysis.append("Exceptional vocabulary diversity")
    elif vocabulary_richness >= 0.5:  # Reduced from 0.6
        vocabulary_score += 15
        vocabulary_analysis.append("Good vocabulary diversity")
    elif vocabulary_richness >= 0.3:  # Reduced from 0.4
        vocabulary_score += 10
        vocabulary_analysis.append("Moderate vocabulary diversity")
    else:
        vocabulary_score -= 5  # Reduced penalty
        vocabulary_analysis.append("Limited vocabulary - consider expanding word choice")
    
    # Advanced word detection (simplified)
    advanced_words = ['nevertheless', 'furthermore', 'consequently', 'subsequently', 
                     'additionally', 'moreover', 'therefore', 'hence', 'thus']
    advanced_word_count = sum(1 for word in words if word in advanced_words)
    
    if advanced_word_count >= 2:  # Reduced from 3
        vocabulary_score += 15
        vocabulary_analysis.append("Excellent use of advanced vocabulary")
    elif advanced_word_count >= 1:
        vocabulary_score += 10
        vocabulary_analysis.append("Good use of advanced vocabulary")
    else:
        vocabulary_score -= 3  # Reduced penalty
        vocabulary_analysis.append("Could benefit from more sophisticated vocabulary")
    
    # Repetition analysis - More lenient
    word_freq = {}
    for word in words:
        if len(word) > 3:  # Only count words longer than 3 characters
            word_freq[word] = word_freq.get(word, 0) + 1
    
    repeated_words = [word for word, count in word_freq.items() if count > 4]  # Increased from 3
    if repeated_words:
        vocabulary_score -= 5  # Reduced penalty
        vocabulary_analysis.append(f"Overuse of words: {', '.join(repeated_words[:3])}")
    else:
        vocabulary_analysis.append("Good word variety - no excessive repetition")
    
    return {
        "score": min(100, max(0, vocabulary_score)),
        "analysis": vocabulary_analysis,
        "metrics": {
            "vocabulary_richness": vocabulary_richness,
            "advanced_word_count": advanced_word_count,
            "repeated_words": len(repeated_words)
        }
    }

def calculate_overall_score(fluency_score, grammar_score, professionalism_score):
    """Calculate overall score with weighted components: 50% fluency, 30% grammar, 20% professionalism"""
    weights = {
        'fluency': 0.50,      # 50% weight for fluency
        'grammar': 0.30,      # 30% weight for grammar
        'professionalism': 0.20  # 20% weight for professionalism
    }
    
    overall_score = (
        fluency_score * weights['fluency'] +
        grammar_score * weights['grammar'] +
        professionalism_score * weights['professionalism']
    )
    
    return round(min(100, overall_score), 1)

def generate_json_report(transcript, fluency_analysis, grammar_analysis, professionalism_analysis, 
                        non_verbal_analysis, vocabulary_analysis, overall_score):
    """Generate the JSON report in the requested format"""
    
    report = {
        "text": transcript,
        "score": overall_score,
        "report": {
            "fluency_analysis": {
                "score": fluency_analysis["score"],
                "analysis": fluency_analysis["analysis"],
                "metrics": fluency_analysis["metrics"]
            },
            "grammar_analysis": {
                "score": grammar_analysis["score"],
                "analysis": grammar_analysis["analysis"],
                "errors": grammar_analysis["errors"],
                "error_count": grammar_analysis["error_count"]
            },
            "professionalism_analysis": {
                "score": professionalism_analysis["score"],
                "analysis": professionalism_analysis["analysis"],
                "metrics": professionalism_analysis["metrics"]
            },
            "non_verbal_cues": {
                "score": non_verbal_analysis["score"],
                "analysis": non_verbal_analysis["analysis"],
                "metrics": non_verbal_analysis["metrics"]
            },
            "vocabulary_analysis": {
                "score": vocabulary_analysis["score"],
                "analysis": vocabulary_analysis["analysis"],
                "metrics": vocabulary_analysis["metrics"]
            }
        }
    }
    
    return report

def main(audio_path, report_path="candidate_report.txt"):
    start_time = time.time()
    print("Transcribing audio...")
    transcript = transcribe_audio(audio_path)
    print("Analyzing fluency...")
    fluency_stats = analyze_fluency(audio_path)
    print("Performing advanced analysis...")
    
    # Perform all analyses
    fluency_analysis = analyze_fluency_advanced(transcript, fluency_stats)
    grammar_analysis = analyze_grammar_advanced(transcript)
    professionalism_analysis = analyze_professionalism(transcript)
    non_verbal_analysis = analyze_non_verbal_cues(fluency_stats, transcript)
    vocabulary_analysis = analyze_vocabulary(transcript)
    
    # Calculate overall score
    overall_score = calculate_overall_score(
        fluency_analysis["score"],
        grammar_analysis["score"],
        professionalism_analysis["score"]
    )
    
    # Generate JSON report
    report = generate_json_report(
        transcript, fluency_analysis, grammar_analysis, professionalism_analysis,
        non_verbal_analysis, vocabulary_analysis, overall_score
    )
    
    # Save human-readable text report
    txt_report_path = report_path.replace('.json', '.txt')
    with open(txt_report_path, "w", encoding="utf-8") as f:
        f.write("=== Audio Analysis Report ===\n\n")
        f.write(f"Overall Score: {overall_score}/100\n\n")
        f.write("Transcribed Text:\n")
        f.write(transcript + "\n\n")
        f.write("=== Detailed Analysis ===\n\n")
        
        f.write("Fluency Analysis:\n")
        f.write(f"Score: {fluency_analysis['score']}/100\n")
        for analysis in fluency_analysis['analysis']:
            f.write(f"- {analysis}\n")
        f.write(f"WPM: {fluency_analysis['metrics']['words_per_minute']:.1f}\n")
        f.write(f"Pause Frequency: {fluency_analysis['metrics']['pause_frequency']:.1f} per minute\n\n")
        
        f.write("Grammar Analysis:\n")
        f.write(f"Score: {grammar_analysis['score']}/100\n")
        for analysis in grammar_analysis['analysis']:
            f.write(f"- {analysis}\n")
        f.write(f"Errors: {grammar_analysis['error_count']}\n\n")
        
        f.write("Professionalism Analysis:\n")
        f.write(f"Score: {professionalism_analysis['score']}/100\n")
        for analysis in professionalism_analysis['analysis']:
            f.write(f"- {analysis}\n")
        f.write(f"Informal Words: {professionalism_analysis['metrics']['informal_words']}\n\n")
        
        f.write("Non-Verbal Cues Analysis:\n")
        f.write(f"Score: {non_verbal_analysis['score']}/100\n")
        for analysis in non_verbal_analysis['analysis']:
            f.write(f"- {analysis}\n")
        f.write(f"Pause Frequency: {non_verbal_analysis['metrics']['pause_frequency']:.1f} per minute\n\n")
        
        f.write("Vocabulary Analysis:\n")
        f.write(f"Score: {vocabulary_analysis['score']}/100\n")
        for analysis in vocabulary_analysis['analysis']:
            f.write(f"- {analysis}\n")
        f.write(f"Vocabulary Richness: {vocabulary_analysis['metrics']['vocabulary_richness']:.2f}\n")
        f.write(f"Advanced Words: {vocabulary_analysis['metrics']['advanced_word_count']}\n")
    
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    print(f"Report saved to {txt_report_path}")
    
    return report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Audio Analysis Model for AI Recruiter")
    parser.add_argument("audio_path", help="Path to the candidate's audio file (wav/mp3)")
    parser.add_argument("--report", default="candidate_report.txt", help="Output report file")
    args = parser.parse_args()
    main(args.audio_path, args.report)
