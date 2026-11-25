"""LoRA fine-tuning of a small instruct model on the JSONL from make_dataset.py (loss on the assistant answer only).
  pip install -r finetune/requirements.txt
  python finetune/train_lora.py --data finetune/data.jsonl --out finetune/adapter
Needs a GPU (a free Colab T4 works with the default 1.5B model). Then run the app with
  CAMPUSSIM_BACKEND=local CAMPUSSIM_ADAPTER=finetune/adapter python app.py"""
import argparse, json, torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct"); ap.add_argument("--data", default="finetune/data.jsonl")
ap.add_argument("--out", default="finetune/adapter"); ap.add_argument("--epochs", type=float, default=2); ap.add_argument("--maxlen", type=int, default=768)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.base)
if tok.pad_token is None: tok.pad_token = tok.eos_token
bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16 if bf16 else torch.float32)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
model.print_trainable_parameters()

class DS(torch.utils.data.Dataset):
    def __init__(s):
        s.items = []
        for line in open(a.data):
            m = json.loads(line)["messages"]
            prompt = tok.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=True)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            f_ids = tok(prompt + m[-1]["content"] + tok.eos_token, add_special_tokens=False, truncation=True, max_length=a.maxlen)["input_ids"]
            s.items.append(dict(input_ids=f_ids, labels=([-100] * len(p_ids) + f_ids[len(p_ids):])[:len(f_ids)]))
    def __len__(s): return len(s.items)
    def __getitem__(s, i): return s.items[i]

def collate(b):
    L = max(len(x["input_ids"]) for x in b); pad = tok.pad_token_id
    return dict(input_ids=torch.tensor([x["input_ids"] + [pad] * (L - len(x["input_ids"])) for x in b]),
                attention_mask=torch.tensor([[1] * len(x["input_ids"]) + [0] * (L - len(x["input_ids"])) for x in b]),
                labels=torch.tensor([x["labels"] + [-100] * (L - len(x["labels"])) for x in b]))

args = TrainingArguments(output_dir=a.out + "-ckpt", per_device_train_batch_size=2, gradient_accumulation_steps=8, num_train_epochs=a.epochs,
                         learning_rate=2e-4, logging_steps=10, save_strategy="no", bf16=bf16, fp16=torch.cuda.is_available() and not bf16,
                         report_to="none", remove_unused_columns=False)
Trainer(model=model, args=args, train_dataset=DS(), data_collator=collate).train()
model.save_pretrained(a.out); tok.save_pretrained(a.out); print("adapter saved to", a.out)
