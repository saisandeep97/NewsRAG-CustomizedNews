import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pinecone import Pinecone
from groq import Groq
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.preprocessing import normalize


# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("GROQ_API_KEY")
INDEX_NAME = 'news-articles'

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Initialize OpenAI
client = Groq(
    api_key=OPENAI_API_KEY,
)

# Load the pre-trained model and tokenizer
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text: str) -> List[float]:
    """Get embedding for a given text using the sentence-transformers model."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return normalize(embeddings.reshape(1, -1))[0].tolist()

def retrieve_relevant_articles(query: str, top_k: int = 5) -> List[Dict]:
    """Retrieve relevant articles from Pinecone based on the query."""
    query_embedding = get_embedding(query)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [result['metadata'] for result in results['matches']]

# ... rest of the code remains the same

def generate_response(query: str, relevant_articles: List[Dict]) -> str:
    """Generate a response using LLM based on the query and relevant articles."""
    context = "\n\n".join([f"Title: {article['title']}\nSource: {article['source']}\nDate: {article['publishedAt']}" 
                           for article in relevant_articles])
    
    prompt = f"""Based on the following news articles, please answer the query in less than 150 tokens: "{query}"

                    Context:
                    {context}

                    Answer:"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on recent news articles."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    #print(response.choices)
    return response.choices[0].message.content.strip()

def rag_news_query(query: str) -> str:
    """Main RAG function to process a news query."""
    relevant_articles = retrieve_relevant_articles(query)
    response = generate_response(query, relevant_articles)
    return response

# Example usage
if __name__ == "__main__":
    sample_queries = [
        "Summarize today's US news in 150 words",
        "What is the latest news in India?",
        "Where is Virat Kohli right now?"
    ]

    for query in sample_queries:
        print(f"Query: {query}")
        response = rag_news_query(query)
        print(f"Response: {response}\n")