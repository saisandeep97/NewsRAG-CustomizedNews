# File: app.py

from groq import Groq
from flask import Flask, render_template, request, jsonify
from rag_news_query import rag_news_query, retrieve_relevant_articles
import os

app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI
client = Groq(
    api_key=OPENAI_API_KEY,
)


def generate_summary(articles, category):
    """Generate a summary of news highlights for a category."""
    context = "\n\n".join([f"Title: {article['title']}\nSource: {article['source']}" 
                           for article in articles[:5]])  # Use top 5 articles
    
    prompt = f"""Summarize the following news highlights for the {category} category in less than 100 words:

{context}

Summary:"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes news highlights."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def query():
    user_query = request.json['query']
    response = rag_news_query(user_query)
    return jsonify({'response': response})

@app.route('/api/highlights')
def highlights():
    categories = ['politics', 'sports', 'business', 'entertainment']
    highlights = {}

    for category in categories:
        articles = retrieve_relevant_articles(f"Latest {category} news", top_k=5)
        summary = generate_summary(articles, category)
        highlights[category] = summary

    return jsonify(highlights)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)