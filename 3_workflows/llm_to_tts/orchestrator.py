"""
LLM to TTS Pipeline Orchestrator

Coordinates the complete workflow between LLM text generation and TTS audio synthesis.
Manages local storage, error handling, and provides a clean interface for pipeline execution.
"""

from pathlib import Path
from typing import List
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
    
    async def process_prompt(self, prompt: str):
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
    
    async def process_batch(self, prompts: List[str]):
        """
        Process multiple prompts through the pipeline.
        
        Args:
            prompts: List of text prompts to process
            
        Returns:
            List of pipeline results for each prompt
        """
        results = []
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n📋 Batch Processing {i}/{len(prompts)}:")
            result = await self.process_prompt(prompt)
            results.append(result)
            
            if result.get("success"):
                print(f"   ✅ Pipeline completed successfully")
            else:
                print(f"   ❌ Pipeline failed: {result.get('error')}")
        
        return results
    
    def _create_error_result(self, error_message: str, prompt: str):
        """Create standardized error result."""
        return {
            "error": error_message,
            "prompt": prompt,
            "success": False,
            "timestamp": datetime.now().isoformat()
        }
    
    def _save_text_result(self, llm_result, prompt: str) -> str:
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
    
    def _save_audio_result(self, tts_result, prompt: str) -> str:
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
    
    def get_pipeline_stats(self, results: List):
        """Generate summary statistics for pipeline execution."""
        successful_runs = [r for r in results if r.get("success")]
        total_words = sum(r["llm_result"]["word_count"] for r in successful_runs)
        
        return {
            "total_prompts": len(results),
            "successful_runs": len(successful_runs),
            "success_rate": len(successful_runs) / len(results) if results else 0,
            "total_words_generated": total_words,
            "output_directory": str(self.output_dir.absolute())
        }