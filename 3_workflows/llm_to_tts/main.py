"""
LLM to TTS Pipeline

OVERVIEW:
This example demonstrates a complete text-to-speech pipeline using Tetra's distributed 
infrastructure. The pipeline consists of two specialized services:

1. LLM Service (A100 GPU): Generates high-quality text content from user prompts
2. TTS Service (RTX 4090 GPU): Converts generated text into natural speech audio

WORKFLOW:
Input Prompt → LLM Generation → Text Processing → TTS Synthesis → Audio Output

USAGE:
Run this script to see the complete pipeline in action with example prompts.
Results are saved locally with timestamped files for easy access.

REQUIREMENTS:
- Tetra library installed
- Local storage for output files
"""

import asyncio

from llm_service import LLMTextGenerator
from tts_service import TTSAudioGenerator
from orchestrator import LLMToTTSOrchestrator


async def main() -> None:
    """Main execution function demonstrating the complete pipeline."""
    print("🎯 LLM to TTS Pipeline with Tetra")
    print("=" * 60)
    print("📋 Pipeline Architecture:")
    print("   • LLM Service  → A100 GPU (Text Generation)")
    print("   • TTS Service  → RTX 4090 GPU (Audio Synthesis)")
    print("   • Local Storage → Organized Results")
    print()
    
    # Initialize services
    print("🔧 Initializing services...")
    llm_service = LLMTextGenerator()
    tts_service = TTSAudioGenerator()
    
    # Create orchestrator
    print("🎼 Creating pipeline orchestrator...")
    orchestrator = LLMToTTSOrchestrator(llm_service, tts_service)
    
    # Demo prompts showcasing different use cases
    demo_prompts = [
        "Explain the importance of renewable energy in simple terms",
        "Tell me an inspiring story about space exploration",
        "Describe the benefits of reading books for personal growth",
        "Share a motivational message for students preparing for exams"
    ]
    
    print(f"\n🎬 Running {len(demo_prompts)} pipeline demonstrations...")
    
    # Process prompts using orchestrator's batch method
    results = await orchestrator.process_batch(demo_prompts)
    
    # Generate and display summary statistics
    print("\n" + "=" * 60)
    print("📊 Pipeline Execution Summary")
    print("=" * 60)
    
    stats = orchestrator.get_pipeline_stats(results)
    
    print(f"✅ Successful runs: {stats['successful_runs']}/{stats['total_prompts']}")
    print(f"📈 Success rate: {stats['success_rate']:.1%}")
    print(f"📝 Total words generated: {stats['total_words_generated']:,}")
    print(f"🎵 Audio files created: {stats['successful_runs']}")
    print(f"📁 Results saved in: {stats['output_directory']}")
    
    if stats['successful_runs'] > 0:
        print(f"\n🎧 Play your generated audio files from:")
        print(f"   {orchestrator.audio_dir.absolute()}")


if __name__ == "__main__":
    asyncio.run(main())