# Chatterbox TTS Inference Example with Tetra

# This example demonstrates how to run text-to-speech generation using Chatterbox TTS
# with the Tetra framework. It showcases:
# - TTS model loading with GPU acceleration
# - Text-to-audio generation from prompts
# - Local audio file saving in WAV format

import asyncio
import io
from pathlib import Path
from tetra_rp import remote, LiveServerless, GpuGroup

gpu_config = LiveServerless(
    gpus=[GpuGroup.AMPERE_80],
    name="example_chatterbox_tts_inference",
    workersMax=1,
)


@remote(
    resource_config=gpu_config,
    dependencies=["chatterbox-tts==0.1.1", "torch", "torchaudio", "setuptools"],
)
class ChatterboxTTSInference:
    def __init__(self):
        import torchaudio as ta
        from chatterbox.tts import ChatterboxTTS
        import io

        self.io = io
        self.ta = ta
        self.model = ChatterboxTTS.from_pretrained(device="cuda")
        print("Chatterbox TTS model loaded successfully!")

    def generate(self, prompt: str):
        """Generate audio from text prompt and return audio data"""
        print(f"Generating audio for: '{prompt}'")

        # Generate audio waveform from the input text
        wav = self.model.generate(prompt)

        # Create buffer and save audio data to return
        buffer = self.io.BytesIO()
        self.ta.save(buffer, wav, self.model.sr, format="wav")
        buffer.seek(0)

        return {
            "prompt": prompt,
            "audio_data": buffer.getvalue(),  # Return raw bytes
            "sample_rate": self.model.sr,
        }


async def main():
    print("🚀 Chatterbox TTS Inference")
    print("=" * 50)

    # Create instance
    print("\n1️⃣ Creating Chatterbox TTS instance...")
    tts = ChatterboxTTSInference()

    # Create output directory locally
    output_dir = Path("audio_outputs")
    output_dir.mkdir(exist_ok=True)

    # Generate audio examples
    prompts = [
        "Hello, this is a test of the Chatterbox TTS system.",
        "The weather today is sunny and beautiful.",
        "Artificial intelligence is transforming our world.",
    ]

    print(f"\n2️⃣ Generating audio for {len(prompts)} prompts:")

    for i, prompt in enumerate(prompts, 1):
        # Generate audio on remote GPU
        result = await tts.generate(prompt)

        # Save audio data locally
        output_path = output_dir / f"output_{i}.wav"
        with open(output_path, "wb") as f:
            f.write(result["audio_data"])

        print(f"   {i}. Generated audio for: '{prompt[:50]}...'")
        print(f"      Saved locally to: {output_path}")

    print(f"\n🎉 Generated {len(prompts)} audio files in {output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
