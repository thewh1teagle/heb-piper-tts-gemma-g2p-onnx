"""
LLM wrapper class for Hebrew text to IPA phoneme conversion using ONNX model.
"""

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM


class Llm:
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.ort_model = ORTModelForCausalLM.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Default system message for Hebrew to IPA conversion
        self.system_message = """Given the following Hebrew sentence, convert it to IPA phonemes.
Input Format: A Hebrew sentence.
Output Format: A string of IPA phonemes.
"""
    
    def create(self, prompt: str) -> str:
        # Prepare conversation with system message and user prompt
        conversation = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template and tokenize
        formatted_prompt = self.tokenizer.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        
        # Generate response
        outputs = self.ort_model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.9,
            top_p=0.95,
            top_k=64,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.convert_tokens_to_ids(["<end_of_turn>", "</s>"])
        )
        
        # Decode without skipping special tokens first to properly extract response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract only the model's response (after the last "model" turn)
        if "<start_of_turn>model" in response:
            response = response.split("<start_of_turn>model")[-1].strip()
            # Remove any end tokens
            for end_token in ["<end_of_turn>", "</s>", "<eos>"]:
                response = response.replace(end_token, "")
        
        return response.strip()

