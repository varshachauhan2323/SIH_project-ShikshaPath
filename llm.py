import json
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class LocalLLM:
    def __init__(self, model_path="./flan_t5_base", dataset_path="career_data.json"):
        # Load dataset
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.df = pd.DataFrame(data)

        # Load embedding model & create FAISS index
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.question_embeddings = self.embedder.encode(self.df['question'].tolist(), convert_to_numpy=True)
        dimension = self.question_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.question_embeddings)

        # Load Flan-T5 model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    def get_response(self, user_question):
        # Find closest question from dataset
        user_embedding = self.embedder.encode([user_question], convert_to_numpy=True)
        D, I = self.index.search(user_embedding, k=1)
        matched_answer = self.df['answer'].iloc[I[0][0]]  # use correct column name

        # -----------------------------
        # Better prompt for Flan-T5
        # -----------------------------
        input_text = f"""
You are a helpful career advisor bot.
Question: {user_question}
Answer: {matched_answer}
Explain in detail in student-friendly language and give examples.
"""

        # Tokenize and generate
        inputs = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=150)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
