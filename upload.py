from transformers import Qwen2ForCausalLM
from transformers import AutoTokenizer
path = "/home/aiops/jiangdf/Workspace/LLaMA-Factory/saves/qwen25_math_openmathreasoning_tir_100K/full/sft"
model = Qwen2ForCausalLM.from_pretrained(path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model.push_to_hub("VerlTool/Qwen2.5-Math-1.5B-TIR-SFT", use_auth_token=True)
tokenizer.push_to_hub("VerlTool/Qwen2.5-Math-1.5B-TIR-SFT", use_auth_token=True)