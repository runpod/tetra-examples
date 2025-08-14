"""
Text-to-Speech Service

High-performance audio synthesis service using Chatterbox TTS on RTX 4090 GPUs.
Converts text to speech with proper error handling and audio data management.
"""

from tetra_rp import remote
from config import get_tts_config


@remote(
    resource_config=get_tts_config(),
    dependencies=["chatterbox-tts==0.1.1", "torch", "torchaudio", "setuptools"],
)
class TTSAudioGenerator:
    """Text-to-Speech service using Chatterbox TTS for high-quality audio synthesis."""
    
    def __init__(self) -> None:
        """Initialize the TTS model and audio processing components."""
        import torchaudio as ta
        from chatterbox.tts import ChatterboxTTS
        import io
        
        self.ta = ta
        self.io = io
        self.model = ChatterboxTTS.from_pretrained(device="cuda")
        
        print("Chatterbox TTS model loaded successfully on RTX 4090!")

    def generate_audio(self, text: str):
        """
        Generate audio from input text.
        
        Args:
            text: Input text to convert to speech
            
        Returns:
            Dictionary containing audio data and metadata
        """
        print(f"Generating audio for: '{text[:50]}...'")
        
        try:
            wav = self.model.generate(text)
            
            buffer = self.io.BytesIO()
            self.ta.save(buffer, wav, self.model.sr, format="wav")
            buffer.seek(0)
            
            audio_data = buffer.getvalue()
            
            print(f"Generated {len(audio_data)} bytes of audio data")
            
            return {
                "text": text,
                "audio_data": audio_data,
                "sample_rate": self.model.sr,
                "file_size_bytes": len(audio_data),
                "service": "tts_generator",
                "success": True
            }
            
        except Exception as e:
            print(f"Error in audio generation: {str(e)}")
            return {
                "error": str(e),
                "text": text,
                "service": "tts_generator",
                "success": False
            }

    def get_model_info(self):
        """Get information about the loaded TTS model."""
        return {
            "model_name": "ChatterboxTTS",
            "sample_rate": self.model.sr,
            "device": "cuda",
            "gpu_type": "RTX 4090"
        }