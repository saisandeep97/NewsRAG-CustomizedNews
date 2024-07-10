import pickle
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import os


# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = 'news-articles'

def initialize_pinecone():
    """Initialize Pinecone client and create index if it doesn't exist."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    if INDEX_NAME not in [index_info["name"] for index_info in pc.list_indexes()]:
        pc.create_index(INDEX_NAME, dimension=384, metric='cosine',spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) )
    
    return pc.Index(INDEX_NAME)

def upload_to_pinecone(index, df):
    """Upload embeddings and metadata to Pinecone."""
    batch_size = 100
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        
        ids = batch.index.astype(str).tolist()
        embeddings = batch['embedding'].tolist()
        batch= batch.fillna('')
        metadata = batch[['title', 'url', 'publishedAt', 'source', 'country', 'category']].to_dict('records')
        to_upsert = list(zip(ids, embeddings, metadata))
        
        index.upsert(vectors=to_upsert)

def main():
    # Load processed data
    with open('news_data/processed_news_articles.pkl', 'rb') as f:
        df = pickle.load(f)
    
    # Initialize Pinecone
    index = initialize_pinecone()
    
    # Upload to Pinecone
    upload_to_pinecone(index, df)
    
    print(f"Uploaded {len(df)} articles to Pinecone index '{INDEX_NAME}'")

if __name__ == "__main__":
    main()