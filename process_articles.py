import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModel
import torch
import re
import pickle


# Load the pre-trained model and tokenizer
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def clean_text(text):
    """Clean and preprocess the text."""
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # We're no longer tokenizing or removing stopwords
        return text
    return ''

def generate_embedding(text):
    """Generate embedding for a given text."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return normalize(embeddings.reshape(1, -1))[0]

def process_articles(input_file):
    """Process articles from CSV file."""
    # Load CSV file
    df = pd.read_csv(input_file)
    
    # Clean text fields
    df['clean_title'] = df['title'].apply(clean_text)
    df['clean_content'] = df['content'].apply(clean_text)
    
    # Combine title and content for embedding
    df['combined_text'] = df['clean_title'] + ' ' + df['clean_content']
    
    # Generate embeddings
    df['embedding'] = df['combined_text'].apply(generate_embedding)
    
    return df

def main():
    input_file = 'news_data/news_articles.csv'  # Replace with your input file name
    output_file = 'news_data/processed_news_articles.pkl'
    
    # Process articles
    processed_df = process_articles(input_file)
    
    # Save processed data
    with open(output_file, 'wb') as f:
        pickle.dump(processed_df, f)
    
    print(f"Processed {len(processed_df)} articles and saved to {output_file}")

if __name__ == "__main__":
    main()