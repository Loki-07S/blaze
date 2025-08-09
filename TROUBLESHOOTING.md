# Troubleshooting Guide

## Common Installation Issues

### 1. Dependency Conflicts

**Problem**: `pandas 2.0.3 requires pytz>=2020.1, but you have pytz 2015.7`

**Solution**:
```bash
# Update pytz first
pip install --upgrade pytz

# Then install requirements
pip install -r requirements.txt
```

### 2. PyTorch Installation Issues

**Problem**: PyTorch fails to install

**Solution**:
```bash
# For CPU-only installation (recommended for most users)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For GPU installation (if you have CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Parselmouth Installation Issues

**Problem**: Parselmouth fails to install on Windows

**Solution**:
```bash
# Install Visual C++ Build Tools first
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Then install parselmouth
pip install parselmouth
```

### 4. Audio Codec Issues

**Problem**: `pydub` can't process audio files

**Solution**:
```bash
# Install ffmpeg
# Windows: Download from https://ffmpeg.org/download.html
# macOS: brew install ffmpeg
# Linux: sudo apt-get install ffmpeg

# Or install pydub with codec support
pip install pydub[ffmpeg]
```

## Step-by-Step Installation

### Method 1: Using the Installation Script (Recommended)

```bash
python install_dependencies.py
```

### Method 2: Manual Installation

```bash
# 1. Upgrade pip
python -m pip install --upgrade pip

# 2. Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Install core dependencies
pip install numpy>=1.21.0 pandas>=1.3.0 pytz>=2020.1

# 4. Install audio processing
pip install openai-whisper>=20231117
pip install parselmouth>=0.4.3
pip install language-tool-python>=2.7.1
pip install pydub>=0.25.1
pip install textblob>=0.17.1
```

### Method 3: Using Conda (Alternative)

```bash
# Create new environment
conda create -n audio_analysis python=3.9
conda activate audio_analysis

# Install dependencies
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install openai-whisper parselmouth language-tool-python pydub textblob
```

## Platform-Specific Issues

### Windows

1. **Visual C++ Build Tools**: Required for some packages
   - Download from Microsoft Visual Studio
   - Install "C++ build tools" workload

2. **FFmpeg**: For audio processing
   - Download from https://ffmpeg.org/download.html
   - Add to PATH environment variable

### macOS

1. **Homebrew**: Install system dependencies
   ```bash
   brew install ffmpeg
   ```

2. **Xcode Command Line Tools**: For compilation
   ```bash
   xcode-select --install
   ```

### Linux

1. **System Dependencies**:
   ```bash
   sudo apt-get update
   sudo apt-get install ffmpeg build-essential
   ```

## Testing Installation

After installation, test if everything works:

```bash
python test_audio_analysis.py
```

## Common Runtime Errors

### 1. "No module named 'whisper'"

**Solution**: Install whisper correctly
```bash
pip install openai-whisper
```

### 2. "No module named 'parselmouth'"

**Solution**: Install parselmouth
```bash
pip install parselmouth
```

### 3. "FFmpeg not found"

**Solution**: Install FFmpeg
- Windows: Download from ffmpeg.org
- macOS: `brew install ffmpeg`
- Linux: `sudo apt-get install ffmpeg`

### 4. "LanguageTool not found"

**Solution**: Install LanguageTool
```bash
pip install language-tool-python
```

## Performance Issues

### 1. Slow Processing

**Solutions**:
- Use GPU if available: Install CUDA version of PyTorch
- Use smaller Whisper model: Change `"tiny"` to `"base"` in code
- Process shorter audio files

### 2. Memory Issues

**Solutions**:
- Use CPU-only PyTorch
- Process audio in smaller chunks
- Close other applications

## Getting Help

If you still have issues:

1. Check your Python version: `python --version`
2. Check pip version: `pip --version`
3. Try creating a virtual environment:
   ```bash
   python -m venv audio_env
   source audio_env/bin/activate  # On Windows: audio_env\Scripts\activate
   ```
4. Share the exact error message for better assistance
