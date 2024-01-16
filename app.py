import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

class InferlessPythonModel:
    def initialize(self):
        base_model_id = "mistralai/Mixtral-8x7B-v0.1"
        peft_model_id = "Tryecho/Mixtral-echo"
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)
        self.ft_model = PeftModel.from_pretrained(base_model,peft_model_id)
        
    def infer(self, inputs):
        prompt = inputs["prompt"]
        model_input = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        self.ft_model.eval()
        with torch.no_grad():
            result = self.tokenizer.decode(self.ft_model.generate(**model_input, max_new_tokens=150, repetition_penalty=1.15)[0], skip_special_tokens=True)
        
        return {'generated_result': result}
        
    def finalize(self):
        pass