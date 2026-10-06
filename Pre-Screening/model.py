# =====================================================
# CPU-compatible Self-Improving MCQ Trainer (LoRA + DPO)
# =====================================================

import json
import random
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

# =====================================================
# CONFIGURATION
# =====================================================
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # small open model
OUTPUT_DIR = Path("./mcq_dpo_model")
BATCH_SIZE = 1  # CPU-friendly
EPOCHS = 1      # small example
MCQ_COUNT = 10000  # target # of MCQs for preference dataset

# =====================================================
# SANITIZED RESUME EXAMPLES
# =====================================================
RESUMES = [
    """
Software Engineer with 3+ years experience in Python, FastAPI, REST APIs,
Docker, AWS, and Machine Learning. Built NLP pipelines and deployed cloud applications.
Strong in data structures, algorithms, and backend development.
""",
    """
Data Scientist with experience in Python, Pandas, Scikit-learn, NLP,
and data visualization. Developed predictive models and deployed on AWS.
"""
]

# =====================================================
# HELPER: GENERATE MCQ (using base model)
# =====================================================
def generate_mcq(model, tokenizer, resume_text):
    prompt = f"""
Generate a multiple choice question (MCQ) from the following resume skills.
- 4 options: A,B,C,D
- One correct answer
- Do NOT mention the candidate
Return JSON like:
{{"question":"text","options":{{"A":"...","B":"...","C":"...","D":"..."}},"correct_answer":"A"}}
Resume: {resume_text}
"""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=150)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    try:
        # Extract JSON from model output
        if "{" in text:
            text = text[text.index("{"):]
        mcq = json.loads(text)
        return mcq
    except:
        return None

# =====================================================
# GENERATE PREFERENCE DATASET (Good vs Bad)
# =====================================================
def build_preference_dataset(model, tokenizer, resumes, mcq_count=1000):
    dataset = []
    for _ in tqdm(range(mcq_count)):
        resume = random.choice(resumes)
        good = None
        # Retry until valid
        while good is None:
            good = generate_mcq(model, tokenizer, resume)
        bad = good.copy()
        # Simple perturbation for 'bad' MCQ
        options = list(bad["options"].values())
        random.shuffle(options)
        bad["options"] = {k: v for k, v in zip(["A","B","C","D"], options)}
        # correct answer may mismatch now
        bad["correct_answer"] = random.choice(["A","B","C","D"])
        dataset.append({"resume": resume, "chosen": good, "rejected": bad})
    return dataset

# =====================================================
# LOAD BASE MODEL & TOKENIZER (CPU)
# =====================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,cache_dir="/tmp/hf_cache")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir="/tmp/hf_cache",
    device_map={"": "cpu"},
    load_in_8bit=True,
    offload_folder="/tmp/model_offload",
    llm_int8_enable_fp32_cpu_offload=True

)

# =====================================================
# APPLY LoRA (small trainable adapters)
# =====================================================
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],  # typical for causal LM
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# =====================================================
# BUILD OR LOAD PREFERENCE DATASET
# =====================================================
dataset_path = Path("preference_dataset.json")
if dataset_path.exists():
    import json
    with open(dataset_path, "r") as f:
        pref_dataset = json.load(f)
else:
    print("Generating preference dataset...")
    pref_dataset = build_preference_dataset(model, tokenizer, RESUMES, MCQ_COUNT)
    with open(dataset_path, "w") as f:
        json.dump(pref_dataset, f, indent=2)

# =====================================================
# CONFIGURE DPO TRAINER
# =====================================================
dpo_config = DPOConfig(
    learning_rate=1e-4,
    batch_size=BATCH_SIZE,
    log_with=None
)

trainer = DPOTrainer(
    model=model,
    tokenizer=tokenizer,
    dataset=pref_dataset,
    **dpo_config.__dict__
)

# =====================================================
# TRAIN (SELF-IMPROVE)
# =====================================================
print("Starting self-improvement training...")
trainer.train()

# =====================================================
# SAVE LoRA WEIGHTS (quantized)
# =====================================================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Self-improved MCQ model saved at {OUTPUT_DIR}")
