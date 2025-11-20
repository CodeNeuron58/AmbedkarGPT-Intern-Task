import os
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_cohere import ChatCohere
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
load_dotenv()

CHUNK_STRATEGIES = {
    "small": (250, 50),
    "medium": (500, 100),
    "large": (900, 150)
}

VECTORSTORE_DIR = "db_{}"
CORPUS_DIR = "corpus"
TEST_DATASET_FILE = "test_dataset.json"
TOP_K = 3  # for retrieval metrics

def load_corpus():
    docs = []
    for i in range(1, 7):
        path = os.path.join(CORPUS_DIR, f"speech{i}.txt")
        if os.path.exists(path):
            loader = TextLoader(path)
            docs.extend(loader.load())
    return docs


def chunk_documents(docs, chunk_size, chunk_overlap):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def build_or_load_vectorstore(chunks, strategy):
    persist_dir = VECTORSTORE_DIR.format(strategy)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_dir)
    return vectordb

def get_llm():
    """Configures and returns the Cohere LLM instance."""
    api = os.getenv("COHERE_API_KEY")
    if not api:
        raise ValueError("COHERE_API_KEY environment variable not set.")
    llm = ChatCohere(model="command-a-03-2025", cohere_api_key=api)
    return llm

def build_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})
    llm = get_llm()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=True
    )
    return qa_chain

# fffffffffffffffffffffffffffffffffff
def normalize(doc):
    return os.path.splitext(os.path.basename(doc))[0].strip()

def hit_rate(retrieved_docs, gold_docs):
    norm_retrieved = [normalize(doc) for doc in retrieved_docs]
    norm_gold = [normalize(doc) for doc in gold_docs]
    return int(any(doc in norm_gold for doc in norm_retrieved))


def mrr(retrieved_docs, gold_docs):
    norm_retrieved = [normalize(doc) for doc in retrieved_docs]
    norm_gold = [normalize(doc) for doc in gold_docs]
    
    for idx, doc in enumerate(norm_retrieved):
        if doc in norm_gold:
            return 1.0 / (idx + 1)
    return 0.0


def precision_at_k(retrieved_docs, gold_docs, k=TOP_K):
    norm_retrieved = [normalize(doc) for doc in retrieved_docs]
    norm_gold = [normalize(doc) for doc in gold_docs]

    retrieved_at_k = norm_retrieved[:k]
    hits = sum(1 for doc in retrieved_at_k if doc in norm_gold)
    return hits / k


def rouge_l_score(prediction, reference):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = scorer.score(reference, prediction)
    return score['rougeL'].fmeasure

def bleu_score(prediction, reference):
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference.split()], prediction.split(), smoothing_function=smoothie)

def cosine_sim(pred_emb, ref_emb):
    return cosine_similarity([pred_emb], [ref_emb])[0][0]

# ----------------------------
# Main evaluation loop
# ----------------------------
def evaluate():
    corpus_docs = load_corpus()

    with open(TEST_DATASET_FILE, "r", encoding="utf-8") as f:
        test_data = json.load(f)["test_questions"]

    results = {}

    for strategy_name, (chunk_size, chunk_overlap) in CHUNK_STRATEGIES.items():
        print(f"\nEvaluating chunking strategy: {strategy_name} ({chunk_size} chars)")
        chunks = chunk_documents(corpus_docs, chunk_size, chunk_overlap)
        vectordb = build_or_load_vectorstore(chunks, strategy_name)
        qa_chain = build_qa_chain(vectordb)

        strategy_results = []

        embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        for q in tqdm(test_data):
            query = q["question"]
            ground_truth = q["ground_truth"]
            source_docs_gold = q["source_documents"]

            try:
                response = qa_chain({"query": query})
                answer = response["result"]
                retrieved_sources = [d.metadata.get("source", "") for d in response["source_documents"]]

                # Retrieval metrics
                hr = hit_rate(retrieved_sources, source_docs_gold)
                rr = mrr(retrieved_sources, source_docs_gold)
                p_at_k = precision_at_k(retrieved_sources, source_docs_gold)

                # Answer quality metrics
                rouge = rouge_l_score(answer, ground_truth)
                bleu = bleu_score(answer, ground_truth)
                try:
                    pred_emb = embeddings_model.embed_query(answer)
                    ref_emb = embeddings_model.embed_query(ground_truth)
                    cosine = cosine_sim(pred_emb, ref_emb)
                except Exception:
                    cosine = None

                strategy_results.append({
                    "id": q["id"],
                    "question": query,
                    "answer": answer,
                    "ground_truth": ground_truth,
                    "retrieved_sources": retrieved_sources,
                    "hit_rate": hr,
                    "MRR": rr,
                    "precision_at_k": p_at_k,
                    "ROUGE-L": rouge,
                    "BLEU": bleu,
                    "CosineSim": cosine
                })
            except Exception as e:
                print(f"Error evaluating question ID {q['id']}: {e}")

        results[strategy_name] = strategy_results

    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\n✅ Evaluation complete! Results saved to test_results.json")

if __name__ == "__main__":
    evaluate()
