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
    """Transcribe audio using whisper (not included in final output)"""
    model = whisper.load_model("tiny")
    result = model.transcribe(audio_path)
    return result['text']

def analyze_fluency_audio_only(audio_path):
    """Analyze fluency using only audio characteristics - no transcription needed"""
    # Use pydub to get audio duration and samples
    audio = AudioSegment.from_file(audio_path)
    duration = len(audio) / 1000.0  # Convert to seconds
    
    # Get audio samples
    samples = np.array(audio.get_array_of_samples())
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))
        # Use average of both channels
        samples = np.mean(samples, axis=1)
    
    # Convert to float and normalize
    samples = samples.astype(np.float32) / np.max(np.abs(samples))
    
    # Calculate audio energy over time
    frame_length = int(0.025 * audio.frame_rate)  # 25ms frames
    hop_length = int(0.010 * audio.frame_rate)    # 10ms hop
    
    energy = []
    for i in range(0, len(samples) - frame_length, hop_length):
        frame = samples[i:i + frame_length]
        energy.append(np.sqrt(np.mean(frame**2)))
    
    energy = np.array(energy)
    
    # Analyze speech activity
    # Find speech segments (high energy) vs silence (low energy)
    energy_threshold = np.percentile(energy, 30)
    speech_segments = energy > energy_threshold
    
    # Calculate speech activity ratio
    speech_activity_ratio = np.sum(speech_segments) / len(speech_segments)
    
    # Find speech rate patterns
    # Count speech bursts (consecutive speech frames)
    speech_bursts = []
    in_speech = False
    burst_start = 0
    
    for i, is_speech in enumerate(speech_segments):
        if is_speech and not in_speech:
            in_speech = True
            burst_start = i
        elif not is_speech and in_speech:
            burst_duration = (i - burst_start) * hop_length / audio.frame_rate
            if burst_duration > 0.1:  # Only count bursts longer than 100ms
                speech_bursts.append(burst_duration)
            in_speech = False
    
    # Calculate speech rate metrics
    total_speech_time = np.sum(speech_bursts)
    speech_rate = total_speech_time / duration if duration > 0 else 0
    
    # Analyze rhythm and flow
    if len(speech_bursts) > 1:
        burst_intervals = np.diff(speech_bursts)
        rhythm_consistency = 1.0 / (1.0 + np.std(burst_intervals))  # Higher is more consistent
    else:
        rhythm_consistency = 0.5  # Default for very short speech
    
    # Analyze energy variation (indicates natural speech patterns)
    energy_variation = np.std(energy[speech_segments]) if np.sum(speech_segments) > 0 else 0
    energy_variation_normalized = energy_variation / np.mean(energy[speech_segments]) if np.mean(energy[speech_segments]) > 0 else 0
    
    # Find pauses and hesitations
    silence_threshold = np.percentile(energy, 15)
    silence_segments = energy < silence_threshold
    
    # Count significant pauses
    pause_count = 0
    in_pause = False
    pause_start = 0
    
    for i, is_silent in enumerate(silence_segments):
        if is_silent and not in_pause:
            in_pause = True
            pause_start = i
        elif not is_silent and in_pause:
            pause_duration = (i - pause_start) * hop_length / audio.frame_rate
            if pause_duration > 0.3:  # Count pauses longer than 300ms
                pause_count += 1
            in_pause = False
    
    # Calculate pause frequency
    pause_frequency = pause_count / (duration / 60) if duration > 0 else 0
    
    # Estimate words per minute based on speech patterns
    # Average English speaker: ~150 WPM, but varies with speech rate
    estimated_wpm = int(speech_rate * 150 * (1 + energy_variation_normalized * 0.5))
    
    return {
        "duration_sec": duration,
        "speech_activity_ratio": speech_activity_ratio,
        "speech_rate": speech_rate,
        "rhythm_consistency": rhythm_consistency,
        "energy_variation": energy_variation_normalized,
        "pause_count": pause_count,
        "pause_frequency": pause_frequency,
        "estimated_wpm": estimated_wpm,
        "speech_bursts": len(speech_bursts)
    }

def analyze_fluency_advanced_audio(fluency_stats):
    """Advanced fluency analysis using audio-only metrics with balanced scoring"""
    duration = fluency_stats['duration_sec']
    speech_activity_ratio = fluency_stats['speech_activity_ratio']
    speech_rate = fluency_stats['speech_rate']
    rhythm_consistency = fluency_stats['rhythm_consistency']
    energy_variation = fluency_stats['energy_variation']
    pause_frequency = fluency_stats['pause_frequency']
    estimated_wpm = fluency_stats['estimated_wpm']
    speech_bursts = fluency_stats['speech_bursts']
    
    # Fluency scoring based on audio characteristics - Balanced scoring
    fluency_score = 0
    fluency_analysis = []
    
    # Speech Activity Analysis (25 points) - More balanced
    if speech_activity_ratio >= 0.7:
        fluency_score += 25
        fluency_analysis.append("Excellent speech activity - good use of speaking time")
    elif speech_activity_ratio >= 0.55:
        fluency_score += 20
        fluency_analysis.append("Good speech activity - reasonable speaking time")
    elif speech_activity_ratio >= 0.4:
        fluency_score += 15
        fluency_analysis.append("Moderate speech activity - some silence")
    elif speech_activity_ratio >= 0.25:
        fluency_score += 10
        fluency_analysis.append("Low speech activity - too much silence")
    else:
        fluency_score += 5
        fluency_analysis.append("Very low speech activity - mostly silence")
    
    # Speech Rate Analysis (25 points) - More balanced ranges
    if 0.65 <= speech_rate <= 0.9:  # Optimal range (more realistic)
        fluency_score += 25
        fluency_analysis.append("Excellent speech rate - natural and engaging pace")
    elif 0.5 <= speech_rate < 0.65 or 0.9 < speech_rate <= 1.1:
        fluency_score += 20
        fluency_analysis.append("Good speech rate - clear and understandable")
    elif 0.35 <= speech_rate < 0.5 or 1.1 < speech_rate <= 1.3:
        fluency_score += 15
        fluency_analysis.append("Moderate speech rate - could be improved")
    elif 0.25 <= speech_rate < 0.35 or 1.3 < speech_rate <= 1.5:
        fluency_score += 10
        fluency_analysis.append("Below average speech rate - needs improvement")
    else:
        fluency_score += 5
        fluency_analysis.append("Poor speech rate - significantly needs improvement")
    
    # Rhythm and Flow Analysis (25 points) - More balanced
    if rhythm_consistency >= 0.7:
        fluency_score += 25
        fluency_analysis.append("Excellent rhythm consistency - smooth flow")
    elif rhythm_consistency >= 0.55:
        fluency_score += 20
        fluency_analysis.append("Good rhythm consistency - generally smooth")
    elif rhythm_consistency >= 0.4:
        fluency_score += 15
        fluency_analysis.append("Moderate rhythm - some irregularity")
    elif rhythm_consistency >= 0.25:
        fluency_score += 10
        fluency_analysis.append("Irregular rhythm - needs improvement")
    else:
        fluency_score += 5
        fluency_analysis.append("Very irregular rhythm - significant improvement needed")
    
    # Pause Analysis (25 points) - More balanced
    if pause_frequency <= 3:
        fluency_score += 25
        fluency_analysis.append("Excellent pause control - minimal hesitation")
    elif pause_frequency <= 6:
        fluency_score += 20
        fluency_analysis.append("Good pause control - reasonable pauses")
    elif pause_frequency <= 10:
        fluency_score += 15
        fluency_analysis.append("Moderate pause control - some hesitation")
    elif pause_frequency <= 15:
        fluency_score += 10
        fluency_analysis.append("Frequent pauses - indicates nervousness")
    else:
        fluency_score += 5
        fluency_analysis.append("Excessive pauses - significant improvement needed")
    
    # Energy Variation Analysis (up to 10 points) - More balanced
    if 0.3 <= energy_variation <= 0.8:  # Natural variation (more realistic)
        fluency_score += 10
        fluency_analysis.append("Natural energy variation - engaging delivery")
    elif 0.15 <= energy_variation < 0.3:
        fluency_score += 7
        fluency_analysis.append("Good energy variation - expressive speech")
    elif 0.8 < energy_variation <= 1.2:
        fluency_score += 5
        fluency_analysis.append("High energy variation - very expressive")
    elif energy_variation > 1.2:
        fluency_score += 3
        fluency_analysis.append("Very high energy variation - overly dramatic")
    else:
        fluency_score += 2
        fluency_analysis.append("Low energy variation - somewhat monotone")
    
    # Duration and Content Bonus - More balanced
    if duration >= 40 and speech_bursts >= 6:  # Higher requirements
        if fluency_score >= 70:
            fluency_score += 5
            fluency_analysis.append("Sustained fluency over extended speech")
        elif fluency_score >= 50:
            fluency_score += 3
            fluency_analysis.append("Good sustained performance")
    elif duration >= 20 and speech_bursts >= 4:  # Higher requirements
        if fluency_score >= 60:
            fluency_score += 2
            fluency_analysis.append("Consistent performance")
    
    # Speech burst analysis - More balanced
    if speech_bursts >= 8:  # Higher requirement
        if rhythm_consistency >= 0.5:  # Higher requirement
            fluency_score += 2
            fluency_analysis.append("Good speech segmentation")
    elif speech_bursts <= 2:  # Penalize more
        fluency_score -= 3  # Higher penalty
        fluency_analysis.append("Limited speech segments - may indicate hesitation")
    
    # Additional factors for more realistic scoring
    if duration < 8:  # Very short speech
        fluency_score += 1  # Reduced bonus
        fluency_analysis.append("Good performance for short speech")
    
    # Cap the score at 100
    fluency_score = min(100, max(0, fluency_score))
    
    return {
        "score": fluency_score,
        "analysis": fluency_analysis,
        "metrics": {
            "speech_activity_ratio": speech_activity_ratio,
            "speech_rate": speech_rate,
            "rhythm_consistency": rhythm_consistency,
            "pause_frequency": pause_frequency,
            "estimated_wpm": estimated_wpm,
            "energy_variation": energy_variation
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

def generate_json_report(fluency_analysis, grammar_analysis, professionalism_analysis, overall_score):
    """Generate the JSON report in the requested format"""
    
    report = {
        "overall_score": overall_score,
        "report": {
            "fluency_analysis": {
                "score": fluency_analysis["score"],
                "analysis": fluency_analysis["analysis"]
            },
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
    
    return report

def analyze_audio(audio_path):
    """Main function to analyze audio and return JSON result"""
    start_time = time.time()
    print("Transcribing audio...")
    transcript = transcribe_audio(audio_path)
    print("Analyzing fluency...")
    fluency_stats = analyze_fluency_audio_only(audio_path)
    print("Performing advanced analysis...")
    
    # Analyze fluency using audio metrics only
    fluency_analysis = analyze_fluency_advanced_audio(fluency_stats)
    
    # Analyze grammar and professionalism using transcribed text
    grammar_analysis = analyze_grammar_advanced(transcript)
    professionalism_analysis = analyze_professionalism(transcript)
    
    # Calculate overall score
    overall_score = calculate_overall_score(
        fluency_analysis["score"],
        grammar_analysis["score"],
        professionalism_analysis["score"]
    )
    
    # Generate JSON report (transcript not included in output)
    report = generate_json_report(
        fluency_analysis, grammar_analysis, professionalism_analysis, overall_score
    )
    
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    
    return report

def analyze_audio_with_text(audio_path, text):
    """Main function to analyze audio with provided text for grammar and professionalism"""
    start_time = time.time()
    print("Analyzing audio and text...")
    
    # Analyze fluency using audio metrics only
    fluency_stats = analyze_fluency_audio_only(audio_path)
    fluency_analysis = analyze_fluency_advanced_audio(fluency_stats)
    
    # Analyze grammar and professionalism using provided text
    grammar_analysis = analyze_grammar_advanced(text)
    professionalism_analysis = analyze_professionalism(text)
    
    # Calculate overall score
    overall_score = calculate_overall_score(
        fluency_analysis["score"],
        grammar_analysis["score"],
        professionalism_analysis["score"]
    )
    
    # Generate JSON report
    report = generate_json_report(
        fluency_analysis, grammar_analysis, professionalism_analysis, overall_score
    )
    
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    
    return report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Audio Analysis Model for AI Recruiter")
    parser.add_argument("audio_path", help="Path to the candidate's audio file (wav/mp3)")
    parser.add_argument("--text", help="Optional transcribed text for grammar and professionalism analysis")
    args = parser.parse_args()
    
    if args.text:
        result = analyze_audio_with_text(args.audio_path, args.text)
    else:
        result = analyze_audio(args.audio_path)
    
    print(json.dumps(result, indent=2))
