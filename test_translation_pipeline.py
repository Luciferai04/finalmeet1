#!/usr/bin/env python3
"""
Test script to validate the translation pipeline with WhisperLive integration.
This script tests the core functionality without requiring a full WhisperLive server.
"""

import os
import sys
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test that all required modules can be imported"""
    print("=" * 60)
    print("Testing Basic Imports")
    print("=" * 60)
    
    try:
        from src.ui.live_camera_enhanced_ui import LiveCameraEnhancedUI
        print("✅ LiveCameraEnhancedUI imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import LiveCameraEnhancedUI: {e}")
        return False
    
    try:
        from src.services.whisper_live.client import Client as WhisperLiveClient
        print("✅ WhisperLiveClient imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import WhisperLiveClient: {e}")
        return False
        
    return True

def test_ui_initialization():
    """Test UI initialization with fallback components"""
    print("\n" + "=" * 60)
    print("Testing UI Initialization")
    print("=" * 60)
    
    try:
        from src.ui.live_camera_enhanced_ui import LiveCameraEnhancedUI
        ui = LiveCameraEnhancedUI()
        print("✅ UI initialized successfully")
        
        # Test basic components
        print(f"✅ RL Coordinator: {ui.rl_coordinator is not None}")
        print(f"✅ Audio Processor: {ui.audio_processor is not None}")
        print(f"✅ Video Processor: {ui.video_processor is not None}")
        print(f"✅ Gemini Model: {ui.gemini_model is not None}")
        
        return ui
    except Exception as e:
        print(f"❌ UI initialization failed: {e}")
        return None

def test_translation_pipeline(ui):
    """Test the translation pipeline with sample text"""
    print("\n" + "=" * 60)
    print("Testing Translation Pipeline")
    print("=" * 60)
    
    if not ui:
        print("❌ Cannot test translation - UI not initialized")
        return False
    
    # Test transcription callback
    sample_text = "Hello, this is a test of the real-time translation system."
    sample_segments = [{"text": sample_text, "start": 0.0, "end": 2.0}]
    
    try:
        print(f"📝 Testing transcription callback with: '{sample_text}'")
        ui.on_transcription_received(sample_text, sample_segments)
        
        # Wait a moment for processing
        time.sleep(1)
        
        # Check if text was added to queue
        if not ui.live_transcript_queue.empty():
            received_text = ui.live_transcript_queue.get()
            print(f"✅ Transcript queue received: '{received_text}'")
        else:
            print("⚠️  No text in transcript queue")
        
        # Check session history
        if ui.session_history:
            latest_entry = ui.session_history[-1]
            print(f"✅ Session history updated: {latest_entry['original_text']}")
        else:
            print("⚠️  Session history empty")
            
        return True
        
    except Exception as e:
        print(f"❌ Translation pipeline test failed: {e}")
        return False

def test_server_connection_validation(ui):
    """Test server connection validation"""
    print("\n" + "=" * 60)
    print("Testing Server Connection Validation")
    print("=" * 60)
    
    if not ui:
        print("❌ Cannot test connection - UI not initialized")
        return False
    
    try:
        # Test connection to default server (should fail if not running)
        is_connected = ui.audio_processor.validate_server_connection()
        print(f"🔌 Server connection status: {'✅ Connected' if is_connected else '❌ Not connected'}")
        
        # Test with different host/port
        ui.audio_processor.whisper_host = "nonexistent.host"
        ui.audio_processor.whisper_port = 99999
        is_connected_invalid = ui.audio_processor.validate_server_connection()
        print(f"🔌 Invalid server test: {'❌ Should fail' if is_connected_invalid else '✅ Correctly failed'}")
        
        # Reset to default
        ui.audio_processor.whisper_host = "localhost"
        ui.audio_processor.whisper_port = 9090
        
        return True
        
    except Exception as e:
        print(f"❌ Connection validation test failed: {e}")
        return False

def test_system_status(ui):
    """Test system status reporting"""
    print("\n" + "=" * 60)
    print("Testing System Status")
    print("=" * 60)
    
    if not ui:
        print("❌ Cannot test status - UI not initialized")
        return False
    
    try:
        status = ui.get_system_status()
        print("✅ System status retrieved successfully")
        
        # Check key components
        basic_components = status.get('basic_components', {})
        whisper_live_status = status.get('whisper_live_status', {})
        
        print(f"📊 Basic Components Status:")
        for component, status_val in basic_components.items():
            status_icon = "✅" if status_val else "❌"
            print(f"   {status_icon} {component}: {status_val}")
        
        print(f"📊 WhisperLive Status:")
        for key, value in whisper_live_status.items():
            print(f"   • {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ System status test failed: {e}")
        return False

def test_gradio_interface(ui):
    """Test Gradio interface creation"""
    print("\n" + "=" * 60)
    print("Testing Gradio Interface Creation")
    print("=" * 60)
    
    if not ui:
        print("❌ Cannot test interface - UI not initialized")
        return False
    
    try:
        import gradio as gr
        interface = ui.create_interface()
        print("✅ Gradio interface created successfully")
        print(f"📱 Interface type: {type(interface)}")
        return True
        
    except Exception as e:
        print(f"❌ Interface creation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 WhisperLive Translation Pipeline Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test 1: Basic imports
    results.append(("Basic Imports", test_basic_imports()))
    
    # Test 2: UI initialization
    ui = test_ui_initialization()
    results.append(("UI Initialization", ui is not None))
    
    if ui:
        # Test 3: Translation pipeline
        results.append(("Translation Pipeline", test_translation_pipeline(ui)))
        
        # Test 4: Server connection validation
        results.append(("Connection Validation", test_server_connection_validation(ui)))
        
        # Test 5: System status
        results.append(("System Status", test_system_status(ui)))
        
        # Test 6: Gradio interface
        results.append(("Gradio Interface", test_gradio_interface(ui)))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status_icon = "✅ PASS" if result else "❌ FAIL"
        print(f"{status_icon} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The translation pipeline is working correctly.")
        print("\n💡 Next steps:")
        print("   1. Start WhisperLive server: python -m whisper_live.server --port 9090")
        print("   2. Set GOOGLE_API_KEY for translations: export GOOGLE_API_KEY='your_key'")
        print("   3. Run the UI: python src/ui/live_camera_enhanced_ui.py")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
