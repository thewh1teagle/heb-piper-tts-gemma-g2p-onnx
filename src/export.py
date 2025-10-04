"""
Export script to convert LoRA-adapted model to ONNX format.
Usage: uv run src/export.py

Upload using:
uv run hf upload --repo-type model thewh1teagle/gemma3-270b-heb-g2p ./gemma3_onnx             
"""

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.exporters.onnx import onnx_export_from_model


def main():
    # Load base model and merge with LoRA adapter
    base_model_id = "google/gemma-3-270m-it"  # The base model for your LoRA
    adapter_id = "thewh1teagle/gemma3-heb-g2p"

    print("Loading base model and LoRA adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
    model = PeftModel.from_pretrained(base_model, adapter_id)
    model = model.merge_and_unload()  # Merge LoRA weights into base model

    tokenizer = AutoTokenizer.from_pretrained(adapter_id)

    # Export merged model to ONNX
    print("Exporting to ONNX...")
    output_dir = "gemma3_onnx"
    onnx_export_from_model(
        model=model,
        output=output_dir,
        task="text-generation-with-past"
    )

    # Save tokenizer to the same directory
    tokenizer.save_pretrained(output_dir)
    print(f"Export complete! Model saved to {output_dir}/")


if __name__ == "__main__":
    main()

