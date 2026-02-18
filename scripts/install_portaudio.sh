#!/bin/bash
# Install PortAudio for desktop audio I/O

set -e

echo "=== Installing PortAudio ==="

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux"
    
    # Try apt (Debian/Ubuntu)
    if command -v apt-get &> /dev/null; then
        echo "Installing PortAudio via apt..."
        sudo apt-get update
        sudo apt-get install -y libportaudio2 libportaudio-dev
    # Try yum (RHEL/CentOS)
    elif command -v yum &> /dev/null; then
        echo "Installing PortAudio via yum..."
        sudo yum install -y portaudio portaudio-devel
    # Try dnf (Fedora)
    elif command -v dnf &> /dev/null; then
        echo "Installing PortAudio via dnf..."
        sudo dnf install -y portaudio portaudio-devel
    else
        echo "Error: Package manager not found. Please install PortAudio manually."
        exit 1
    fi
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS"
    
    if command -v brew &> /dev/null; then
        echo "Installing PortAudio via Homebrew..."
        brew install portaudio
    else
        echo "Error: Homebrew not found. Please install PortAudio manually:"
        echo "  brew install portaudio"
        exit 1
    fi
    
else
    echo "Error: Unsupported OS: $OSTYPE"
    echo "Please install PortAudio manually for your system."
    exit 1
fi

echo ""
echo "PortAudio installation complete!"
echo ""
echo "To verify:"
echo "  pkg-config --modversion portaudio-2.0"
