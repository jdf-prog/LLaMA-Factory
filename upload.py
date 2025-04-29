from transformers import Qwen2ForCausalLM
from transformers import AutoTokenizer
path = "saves/qwen25_interpreter_thinking_tool/full/sft/checkpoint-444"
model = Qwen2ForCausalLM.from_pretrained(path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model.push_to_hub("VerlTool/Qwen2.5-Coder-7B-Inst-Interpreter-thinking-valid-tool", use_auth_token=True)
tokenizer.push_to_hub("VerlTool/Qwen2.5-Coder-7B-Inst-Interpreter-thinking-valid-tool", use_auth_token=True)