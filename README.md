# AI-powered Banking Chatbot

This project is an intelligent banking chatbot built with Flask backend. It uses a fine-tuned DistilBERT model to detect user intents (loan application, credit card application, general inquiries), performs sentiment analysis on user inputs, dynamically generates PDF application forms, and interacts with a Mistral large language model via the Ollama API.

## Key Features

- User intent detection with a custom fine-tuned DistilBERT model.
- Sentiment analysis using TextBlob.
- Dynamic PDF form generation for loan and credit card applications.
- Interaction with Mistral LLM via Ollama API for natural language responses.
- Session management with form state tracking.
- Input validation to prevent malicious content.
- Detailed logging of conversations and errors.

## Technologies Used

- Python 3.x
- Flask (with Flask-Session and Flask-CORS)
- PyTorch & Transformers (HuggingFace DistilBERT)
- TextBlob (sentiment analysis)
- ReportLab (PDF generation)
- Requests (API calls)
- Ollama API (Mistral LLM integration)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/EyaDorai/BankBot.git
   
