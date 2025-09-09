from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-base"

# Download and save locally
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

tokenizer.save_pretrained("flan_t5_base")
model.save_pretrained("flan_t5_base")

print("✅ Model downloaded and saved locally in 'flan_t5_base'")
