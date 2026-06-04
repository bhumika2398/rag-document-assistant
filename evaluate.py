from rag_pipeline import load_and_store_pdf
from agent import create_agent, ask_agent
from dotenv import load_dotenv

load_dotenv()

test_pairs = [
    {
        "question": "What is the Transformer model?",
        "expected_keywords": ["attention", "encoder", "decoder", "architecture"]
    },
    {
        "question": "What is self-attention?",
        "expected_keywords": ["positions", "sequence", "representation", "intra-attention"]
    },
    {
        "question": "What is multi-head attention?",
        "expected_keywords": ["jointly", "subspaces", "positions", "concatenated"]
    },
    {
        "question": "What BLEU score did the model achieve on English to French?",
        "expected_keywords": ["41", "french", "bleu", "translation"]
    },
    {
        "question": "How many attention heads does the Transformer use?",
        "expected_keywords": ["8", "eight", "heads", "parallel"]
    },
    {
        "question": "What optimizer was used to train the model?",
        "expected_keywords": ["adam", "optimizer", "beta", "learning rate"]
    },
    {
        "question": "What is the encoder in the Transformer?",
        "expected_keywords": ["stack", "layers", "self-attention", "feed-forward"]
    },
    {
        "question": "What is dropout and how was it used?",
        "expected_keywords": ["dropout", "0.1", "sub-layer", "normalized"]
    },
    {
        "question": "What is positional encoding?",
        "expected_keywords": ["position", "sequence", "embeddings", "encoding"]
    },
    {
        "question": "Why did the authors propose the Transformer?",
        "expected_keywords": ["recurrence", "parallelization", "attention", "convolutions"]
    }
]

def evaluate_faithfulness(answer, context):
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())
    overlap = len(answer_words.intersection(context_words))
    faithfulness = min(overlap / max(len(answer_words), 1), 1.0)
    return round(faithfulness, 2)

def evaluate_relevancy(answer, keywords):
    answer_lower = answer.lower()
    matched = sum(1 for kw in keywords if kw.lower() in answer_lower)
    relevancy = matched / len(keywords)
    return round(relevancy, 2)

def run_evaluation():
    print("Loading PDF and creating agent...")
    vectorstore = load_and_store_pdf("test.pdf")
    agent = create_agent(vectorstore)

    faithfulness_scores = []
    relevancy_scores = []

    print("\nRunning 10 test questions...\n")

    for i, pair in enumerate(test_pairs):
        print(f"Q{i+1}: {pair['question'][:60]}...")
        answer, _ = ask_agent(agent, pair["question"], [])
        docs = vectorstore.similarity_search(pair["question"], k=5)
        context = " ".join([d.page_content for d in docs])

        f_score = evaluate_faithfulness(answer, context)
        r_score = evaluate_relevancy(answer, pair["expected_keywords"])

        faithfulness_scores.append(f_score)
        relevancy_scores.append(r_score)
        print(f"   Faithfulness: {f_score}  |  Relevancy: {r_score}")

    avg_faithfulness = round(sum(faithfulness_scores) / len(faithfulness_scores), 2)
    avg_relevancy = round(sum(relevancy_scores) / len(relevancy_scores), 2)

    print("\n" + "="*45)
    print("EVALUATION RESULTS — 10 test questions")
    print("="*45)
    print(f"Faithfulness Score:      {avg_faithfulness}")
    print(f"Answer Relevancy Score:  {avg_relevancy}")
    print("="*45)
    print("\nFaithfulness: measures how grounded answers are")
    print("in retrieved document context (1.0 = perfect)")
    print("Answer Relevancy: measures how well answers")
    print("address the question asked (1.0 = perfect)")

if __name__ == "__main__":
    run_evaluation()