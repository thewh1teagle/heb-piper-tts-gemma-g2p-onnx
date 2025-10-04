"""
wget https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/shaul.onnx
wget https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/model.config.json
Usage: uv run src/main.py
"""

from llm import Llm
from piper_onnx import Piper
import soundfile as sf
import huggingface_hub


# Initialize the LLM with the exported model
model_id = 'thewh1teagle/gemma3-270b-heb-g2p'
llm_model_dir = 'gemma3_onnx'
llm_model_dir = huggingface_hub.snapshot_download(model_id, local_dir=llm_model_dir)
tts_model_path, tts_config_path = "shaul.onnx", "model.config.json"

prompt = "הכוח לשנות מתחיל ברגע שבו אתה מאמין שזה אפשרי!"
output_path = "audio.wav"

llm = Llm(llm_model_dir)
tts = Piper(tts_model_path, tts_config_path)

# Generate response
response = llm.create(prompt)
samples, sample_rate = tts.create(response, is_phonemes=True)
sf.write(output_path, samples, sample_rate)
print(f"Created {output_path}")