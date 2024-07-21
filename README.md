
# NewsInsight AI

## Overview

This project is a News Retrieval-Augmented Generation (RAG) system that fetches top news articles from various sources, processes them, and allows users to query this information using natural language. The system uses advanced NLP techniques to provide accurate and up-to-date responses to news-related queries.

## Features

- Automated news collection from multiple sources and categories
- Text preprocessing and embedding generation
- Vector database storage for efficient retrieval
- RAG-based query answering system
- Web interface for user interactions

## Technology Stack

- Python 3.8+
- Flask (Web Framework)
- OpenAI API (for embeddings and text generation)
- Pinecone (Vector Database)
- NewsAPI (for fetching news articles)

## Setup and Installation

### Prerequisites

- Python 3.8+
- NewsAPI Key
- OpenAI API Key
- Pinecone API Key

### Local Setup

1. Clone the repository:
   ```
   git clone https://github.com/saisandeep97/NewsRAG-CustomizedNews.git
   cd news-rag-system
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the root directory and add the following:
   ```
   NEWS_API_KEY=your_newsapi_key
   OPENAI_API_KEY=your_openai_key
   PINECONE_API_KEY=your_pinecone_key
   PINECONE_ENVIRONMENT=your_pinecone_environment
   ```

5. Run the application:
   ```
   python app.py
   ```

## Usage

### Data Collection and Processing

To manually run the data collection and processing pipeline:

```
python news_collector.py
python process_articles.py
python upload_to_pinecone.py
```

### Web Interface

After starting the Flask application, navigate to `http://localhost:5000` in your web browser. You can enter news-related queries in the provided input field and view the generated responses.


## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## Acknowledgments

- NewsAPI for providing access to news articles
- OpenAI for their powerful language models and embeddings
- Pinecone for vector similarity search capabilities
