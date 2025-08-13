# Live Camera Enhanced Translator - Usage Guide

## Quick Start

### Option 1: Using the Startup Script (Recommended)
```bash
./start_translator.sh
```

### Option 2: Manual Setup
```bash
# 1. Set API Key
export GOOGLE_API_KEY="AIzaSyAO8JzfpF9NpPbgQWCm40wJEbMvuplsA5A"

# 2. Start WhisperLive Server
python3 WhisperLive/simple_whisper_server.py --port 9090 &

# 3. Start Main Application
python3 live_camera_enhanced_ui.py
```

## Features Overview

### **Live Camera Feed**
- Real-time webcam video stream
- Browser-based camera access
- Interactive controls

### **Real-Time Audio Processing**
- WhisperLive server integration
- Voice Activity Detection (VAD)
- Real-time transcription to English

### **Live Translation**
- **Bengali (bn)** - বাংলা অনুবাদ
- **Hindi (hi)** - हिंदी अনुवাद
- Powered by Google Gemini AI
- Real-time translation display

### **Advanced Features**
- Topic analysis and schema checking
- Metacognitive optimization
- EgoSchema integration
- Session recording and transcripts

## **User Interface**

### Main Tab: "Live Translation (WhisperLive)"
```
┌─────────────────┐    ┌──────────────────────────────┐
│  Camera Feed    │    │  Live Transcript              │
│  Audio Status   │    │  (English text appears here) │
│                 │    │                              │
│  Controls:      │    │  Live Translation             │
│  • Class ID     │    │  (Bengali/Hindi text here)   │
│  • Language     │    │                              │
│  • Server       │    │                              │
│  • Start/Stop   │    │                              │
└─────────────────┘    └──────────────────────────────┘
```

### Other Tabs:
- **Topic Analysis**: Upload course materials and analyze coverage
- **System Status**: Monitor all components and performance
- **Demo Schema Generator**: Create sample course materials

## **Step-by-Step Usage**

### 1. **Launch Application**
```bash
./start_translator.sh
```
- Wait for "Server started successfully" message
- Open browser to http://127.0.0.1:7860

### 2. **Grant Permissions**
- **Camera**: Allow webcam access when prompted
- **Microphone**: Grant microphone permissions for audio

### 3. **Configure Session**
- **Class ID**: Enter a session identifier (e.g., "MATH101")
- **Target Language**: Select Bengali (bn) or Hindi (hi)
- **Server**: Keep default "localhost:9090"

### 4. **Start Live Session**
- Click "**Start Live Session**" button
- Wait for "Live session started" confirmation
- Camera feed should appear
- Audio status should show "Ready"

### 5. **Begin Translation**
- **Speak clearly** into your microphone
- Watch for English text in "Live Transcript"
- Translated text appears in "Live Translation"
- Both update in real-time

### 6. **Stop Session**
- Click "**Stop Session**" when finished
- Transcript is automatically saved
- Schema is generated for analysis

## **Language Configuration**

### Bengali (bn)
```
English: "Hello, how are you?"
Bengali: "নমস্কার, আপনি কেমন আছেন?"
```

### Hindi (hi)
```
English: "Hello, how are you?"
Hindi: "नमस्ते, आप कैसे हैं?"
```

## **Troubleshooting**

### WhisperLive Server Issues
```
[ERROR] Failed to connect to WhisperLive server
```
**Solution**: Restart the server
```bash
pkill -f simple_whisper_server
python3 WhisperLive/simple_whisper_server.py --port 9090 &
```

### Translation Not Working
```
[WARNING] No Gemini model available, using fallback
```
**Solution**: Check API key is set
```bash
echo $GOOGLE_API_KEY
# Should output: AIzaSyAO8JzfpF9NpPbgQWCm40wJEbMvuplsA5A
```

### Camera/Audio Permissions
- **Chrome**: Click lock icon → Camera/Microphone → Allow
- **Firefox**: Click shield icon → Permissions → Allow
- **Safari**: Safari → Preferences → Websites → Camera/Microphone

### Port Already in Use
```
[ERROR] Address already in use: localhost:9090
```
**Solution**: Kill existing processes
```bash
lsof -ti:9090 | xargs kill -9
```

## **System Status**

Check component status in the "System Status" tab:

### Basic Components
- **whisper_model**: Loaded
- **gemini_model**: Connected
- **whisper_live_client**: Active
- **audio_processor**: Running
- **video_processor**: Capturing

### Advanced Features
- **egoschema_integration**: Available
- **metacognitive_controller**: Active
- **performance_monitor**: Tracking

## **Use Cases**

### 1. **Live Classroom Translation**
- Teacher speaks in English → Students see Bengali/Hindi
- Real-time lecture translation
- Session recording for review

### 2. **Meeting Translation**
- International team meetings
- Real-time conversation translation
- Multi-language support

### 3. **Educational Content**
- Online course translation
- Tutorial real-time subtitles
- Language learning assistance

## **Performance Tips**

### Optimal Setup
- **Internet**: Stable connection for Gemini API
- **Hardware**: Good microphone for clear audio
- **Browser**: Chrome/Firefox recommended
- **Lighting**: Good lighting for camera feed

### Quality Settings
- Speak clearly and at moderate pace
- Minimize background noise
- Position microphone appropriately
- Ensure stable internet connection

## **Security Notes**

- API key is configured locally
- No data stored on external servers
- All processing happens in real-time
- Session transcripts saved locally only

---

## **Support**

If you encounter issues:
1. Check the console output for error messages
2. Verify all components are running (System Status tab)
3. Restart the application if needed
4. Ensure all dependencies are installed

**Happy translating!**
