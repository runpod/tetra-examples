"""
Shared Configuration for LLM to TTS Pipeline

Centralized configuration management for network volumes and service settings.
"""

from tetra_rp import LiveServerless, GpuGroup, NetworkVolume


# Shared network storage
SHARED_STORAGE = NetworkVolume(name="example_llm_tts_storage", size=100)


def get_llm_config() -> LiveServerless:
    """
    Get configuration for LLM text generation service.
    
    Returns:
        LiveServerless configuration optimized for text generation
    """
    return LiveServerless(
        gpus=[GpuGroup.AMPERE_80],
        name="example_llm_text_generator",
        networkVolume=SHARED_STORAGE,
        workersMax=1
    )


def get_tts_config() -> LiveServerless:
    """
    Get configuration for TTS audio synthesis service.
    
    Returns:
        LiveServerless configuration optimized for audio generation
    """
    return LiveServerless(
        gpus=[GpuGroup.ADA_24],
        name="example_tts_audio_generator", 
        networkVolume=SHARED_STORAGE,
        workersMax=1
    )