import os
import time
import pandas as pd
from dotenv import load_dotenv
from rag_pipeline import setup_rag_chain
from langchain_core.messages import HumanMessage

load_dotenv()

INDEX_NAME = "rag-pdf-index"
NAMESPACE = "assignment-namespace"
MODEL_NAME = "openai/gpt-oss-20b"
TOP_K = 4

# Test Dataset for Evaluation
TEST_DATASET = [
    {
        "query": "What is the main topic of the document?",
        "type": "Summarization"
    },
    {
        "query": "What is mentioned on page 1?",
        "type": "Page-Specific"
    },
    {
        "query": "List the key keypoints or sections in the text.",
        "type": "Keypoint Extraction"
    },
    {
        "query": "Is there any information about advanced configurations?",
        "type": "Specific Inquiry"
    }
]

def run_evaluation():
    print("🚀 Initializing Evaluation Pipeline...")
    
    if not os.getenv("PINECONE_API_KEY") or not os.getenv("GROQ_API_KEY"):
        print("❌ Error: API Keys missing in .env file!")
        return

    try:
        rag_chain, vectorstore = setup_rag_chain(INDEX_NAME, MODEL_NAME, TOP_K, NAMESPACE)
    except Exception as e:
        print(f"❌ Error setting up RAG chain: {e}")
        return

    results = []

    for idx, test_item in enumerate(TEST_DATASET, start=1):
        query = test_item["query"]
        q_type = test_item["type"]
        
        print(f"\n[Test {idx}/{len(TEST_DATASET)}] Query: '{query}'")
        
        start_time = time.time()
        try:
            response = rag_chain.invoke({"input": query, "chat_history": []})
            latency = round(time.time() - start_time, 2)
            
            answer = response.get("answer", "")
            retrieved_docs = response.get("context", [])
            num_chunks = len(retrieved_docs)
            
            is_fallback = "The answer is not available in the provided document." in answer
            status = "Fallback / Not Found" if is_fallback else "Success"
            
            results.append({
                "Test ID": idx,
                "Query Type": q_type,
                "Query": query,
                "Latency (s)": latency,
                "Retrieved Chunks": num_chunks,
                "Status": status,
                "Answer Preview": answer[:120].replace("\n", " ") + "..."
            })
            
            print(f"   ⏱️ Latency: {latency}s | Chunks Retrieved: {num_chunks} | Status: {status}")

        except Exception as e:
            print(f"   ❌ Execution failed: {e}")

    # Generate Evaluation Summary Table
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("📊 RAG EVALUATION BENCHMARK REPORT")
    print("="*80)
    print(df_results[["Test ID", "Query Type", "Latency (s)", "Retrieved Chunks", "Status"]].to_string(index=False))
    
    # Save Benchmark Output
    csv_filename = "rag_evaluation_report.csv"
    df_results.to_csv(csv_filename, index=False)
    print(f"\n✅ Full report successfully exported to `{csv_filename}`!")

if __name__ == "__main__":
    run_evaluation()