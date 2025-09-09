from local_llm.local_llm import LocalLLM

# Initialize the model (loads Flan-T5 + embeddings)
print("Loading model and embeddings...")
llm = LocalLLM()
print("Model loaded successfully!")

# Test a few questions
test_questions = [
    "What can I do after 12th science?",
    "Which career is good after 12th commerce?",
    "What are options after 12th arts?"
]

for q in test_questions:
    answer = llm.get_response(q)
    print(f"Q: {q}")
    print(f"A: {answer}\n")
