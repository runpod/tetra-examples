"""
LLM to TTS Pipeline - Main Orchestrator

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
from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime

from llm_service import LLMTextGenerator
from tts_service import TTSAudioGenerator


class LLMToTTSOrchestrator:
    """
    Main orchestrator for the LLM to TTS pipeline.
    
    Coordinates between LLM text generation and TTS audio synthesis services,
    manages local storage, and provides a complete text-to-speech workflow.
    """
    
    def __init__(self, llm_service: LLMTextGenerator, tts_service: TTSAudioGenerator) -> None:
        """
        Initialize the pipeline orchestrator.
        
        Args:
            llm_service: Initialized LLM text generation service
            tts_service: Initialized TTS audio generation service
        """
        self.llm_service = llm_service
        self.tts_service = tts_service
        
        self._setup_output_directories()
        
        print("LLM to TTS Pipeline Orchestrator initialized")
        print(f"Results directory: {self.output_dir.absolute()}")
    
    def _setup_output_directories(self) -> None:
        """Create organized output directory structure."""
        self.output_dir = Path("llm_tts_results")
        self.output_dir.mkdir(exist_ok=True)
        
        self.texts_dir = self.output_dir / "texts"
        self.audio_dir = self.output_dir / "audio"
        
        self.texts_dir.mkdir(exist_ok=True)
        self.audio_dir.mkdir(exist_ok=True)
    
    async def process_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Execute the complete LLM to TTS pipeline for a single prompt.
        
        Args:
            prompt: Input text prompt for processing
            
        Returns:
            Complete pipeline result with file paths and metadata
        """
        print(f"\n🚀 Processing: '{prompt}'")
        
        # Step 1: Generate text with LLM
        print("   📝 Step 1: Generating text content...")
        llm_result = await self.llm_service.generate_text(prompt)
        
        if not llm_result.get("success"):
            return self._create_error_result(f"LLM failed: {llm_result.get('error')}", prompt)
        
        generated_text = llm_result["generated_text"]
        print(f"   ✅ Generated {llm_result['word_count']} words")
        
        # Step 2: Convert text to speech
        print("   🎵 Step 2: Converting to speech...")
        tts_result = await self.tts_service.generate_audio(generated_text)
        
        if not tts_result.get("success"):
            return self._create_error_result(f"TTS failed: {tts_result.get('error')}", prompt)
        
        print(f"   ✅ Generated {tts_result['file_size_bytes']} bytes of audio")
        
        # Step 3: Save results locally
        print("   💾 Step 3: Saving results...")
        text_file = self._save_text_result(llm_result, prompt)
        audio_file = self._save_audio_result(tts_result, prompt)
        
        result = {
            "prompt": prompt,
            "llm_result": llm_result,
            "tts_result": tts_result,
            "saved_files": {
                "text_file": text_file,
                "audio_file": audio_file
            },
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"   ✅ Saved: {Path(text_file).name} | {Path(audio_file).name}")
        return result
    
    def _create_error_result(self, error_message: str, prompt: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            "error": error_message,
            "prompt": prompt,
            "success": False,
            "timestamp": datetime.now().isoformat()
        }
    
    def _save_text_result(self, llm_result: Dict[str, Any], prompt: str) -> str:
        """Save generated text with metadata to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"text_{timestamp}.json"
        filepath = self.texts_dir / filename
        
        text_data = {
            "timestamp": timestamp,
            "original_prompt": prompt,
            "generated_text": llm_result["generated_text"],
            "word_count": llm_result["word_count"],
            "text_length": llm_result["text_length"],
            "service_info": {
                "service": llm_result["service"],
                "gpu_type": llm_result["gpu_type"]
            }
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(text_data, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def _save_audio_result(self, tts_result: Dict[str, Any], prompt: str) -> str:
        """Save generated audio file with metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create safe filename from prompt
        safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_prompt = safe_prompt.replace(' ', '_') or "audio"
        
        # Save WAV file
        audio_filename = f"speech_{safe_prompt}_{timestamp}.wav"
        audio_filepath = self.audio_dir / audio_filename
        
        with open(audio_filepath, "wb") as f:
            f.write(tts_result["audio_data"])
        
        # Save metadata
        metadata_filename = f"speech_info_{safe_prompt}_{timestamp}.json"
        metadata_filepath = self.audio_dir / metadata_filename
        
        audio_metadata = {
            "timestamp": timestamp,
            "original_prompt": prompt,
            "text_content": tts_result["text"],
            "sample_rate": tts_result["sample_rate"],
            "file_size_bytes": tts_result["file_size_bytes"],
            "audio_file": audio_filename,
            "service": tts_result["service"]
        }
        
        with open(metadata_filepath, "w", encoding="utf-8") as f:
            json.dump(audio_metadata, f, indent=2, ensure_ascii=False)
        
        return str(audio_filepath)


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
    
    results: List[Dict[str, Any]] = []
    
    for i, prompt in enumerate(demo_prompts, 1):
        print(f"\n📋 Demo {i}/{len(demo_prompts)}:")
        
        result = await orchestrator.process_prompt(prompt)
        results.append(result)
        
        if result.get("success"):
            print(f"   ✅ Pipeline completed successfully")
        else:
            print(f"   ❌ Pipeline failed: {result.get('error')}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Pipeline Execution Summary")
    print("=" * 60)
    
    successful_runs = [r for r in results if r.get("success")]
    total_words = sum(r["llm_result"]["word_count"] for r in successful_runs)
    
    print(f"✅ Successful runs: {len(successful_runs)}/{len(demo_prompts)}")
    print(f"📝 Total words generated: {total_words:,}")
    print(f"🎵 Audio files created: {len(successful_runs)}")
    print(f"📁 Results saved in: {orchestrator.output_dir.absolute()}")
    
    if successful_runs:
        print(f"\n🎧 Play your generated audio files from:")
        print(f"   {orchestrator.audio_dir.absolute()}")


if __name__ == "__main__":
    asyncio.run(main())