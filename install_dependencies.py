#!/usr/bin/env python3
"""
Installation script for Audio Analysis Model dependencies
This script handles the installation of all required packages with proper error handling
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def upgrade_pip():
    """Upgrade pip to latest version"""
    return run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")

def install_torch():
    """Install PyTorch with appropriate version"""
    print("\n🔄 Installing PyTorch...")
    
    # Try CPU version first (more compatible)
    if run_command(f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu", "Installing PyTorch (CPU version)"):
        return True
    
    # Fallback to regular torch
    return run_command(f"{sys.executable} -m pip install torch", "Installing PyTorch (fallback)")

def install_core_dependencies():
    """Install core dependencies"""
    dependencies = [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "pytz>=2020.1",
        "textblob>=0.17.1",
        "pydub>=0.25.1"
    ]
    
    for dep in dependencies:
        if not run_command(f"{sys.executable} -m pip install {dep}", f"Installing {dep}"):
            return False
    return True

def install_audio_dependencies():
    """Install audio processing dependencies"""
    audio_deps = [
        "openai-whisper>=20231117",
        "parselmouth>=0.4.3",
        "language-tool-python>=2.7.1"
    ]
    
    for dep in audio_deps:
        if not run_command(f"{sys.executable} -m pip install {dep}", f"Installing {dep}"):
            return False
    return True

def test_imports():
    """Test if all modules can be imported"""
    print("\n🔄 Testing imports...")
    
    test_modules = [
        "whisper",
        "parselmouth", 
        "numpy",
        "language_tool_python",
        "pydub",
        "torch",
        "textblob"
    ]
    
    failed_imports = []
    
    for module in test_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("✅ All imports successful!")
    return True

def main():
    """Main installation process"""
    print("🚀 Audio Analysis Model - Dependency Installation")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Upgrade pip
    if not upgrade_pip():
        print("⚠️  Warning: pip upgrade failed, continuing...")
    
    # Install PyTorch first (can be problematic)
    if not install_torch():
        print("❌ PyTorch installation failed")
        return False
    
    # Install core dependencies
    if not install_core_dependencies():
        print("❌ Core dependencies installation failed")
        return False
    
    # Install audio dependencies
    if not install_audio_dependencies():
        print("❌ Audio dependencies installation failed")
        return False
    
    # Test imports
    if not test_imports():
        print("❌ Import test failed")
        return False
    
    print("\n🎉 Installation completed successfully!")
    print("\nYou can now run the audio analysis model:")
    print("python audio_analysis_v2.py your_audio_file.wav")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Installation failed. Please check the errors above.")
        sys.exit(1)
