"""
Export script to convert LoRA-adapted model to ONNX format.
Usage: 
  uv run src/export.py              # Export unquantized (fp32)
  uv run src/export.py --quantize   # Export fp32 then quantize to int8

Upload using:
uv run hf upload --repo-type model thewh1teagle/gemma3-270b-heb-g2p ./gemma3_onnx             
"""

import argparse
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.exporters.onnx import onnx_export_from_model
from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig


def quantize(model_dir: str):
    """
    Apply int8 dynamic quantization to an ONNX model.
    
    Args:
        model_dir: Directory containing the ONNX model to quantize (will be modified in-place)
    """
    print(f"Quantizing model in {model_dir}...")
    
    # Load the ONNX model
    ort_model = ORTModelForCausalLM.from_pretrained(model_dir)
    
    # Create quantizer and config
    quantizer = ORTQuantizer.from_pretrained(ort_model)
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    
    # Quantize and save back to the same directory
    quantizer.quantize(save_dir=model_dir, quantization_config=qconfig)
    
    # Show model size after quantization
    model_size = sum(f.stat().st_size for f in Path(model_dir).rglob("*.onnx")) / (1024**2)
    print(f"Quantized model size: {model_size:.1f} MB")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Export LoRA model to ONNX")
    parser.add_argument(
        "--quantize", 
        action="store_true", 
        help="Apply int8 quantization after export"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="gemma3_onnx",
        help="Output directory for the exported model"
    )
    args = parser.parse_args()
    
    # Load base model and merge with LoRA adapter
    base_model_id = "google/gemma-3-270m-it"
    adapter_id = "thewh1teagle/gemma3-heb-g2p"

    print("Loading base model and LoRA adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
    model = PeftModel.from_pretrained(base_model, adapter_id)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(adapter_id)

    # Export to ONNX (fp32)
    print("Exporting to ONNX...")
    output_dir = args.output_dir
    onnx_export_from_model(
        model=model,
        output=output_dir,
        task="text-generation-with-past"
    )
    tokenizer.save_pretrained(output_dir)
    
    # Show model size
    model_size = sum(f.stat().st_size for f in Path(output_dir).rglob("*.onnx")) / (1024**2)
    print(f"Export complete! Model saved to {output_dir}/")
    print(f"Model size: {model_size:.1f} MB")
    
    # Optionally quantize
    if args.quantize:
        quantize(output_dir)


if __name__ == "__main__":
    main()
