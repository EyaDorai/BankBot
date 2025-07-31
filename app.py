import os
import logging
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file, session
from flask_cors import CORS
from flask_session import Session
from dotenv import load_dotenv
import requests
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from typing import Dict, Any, Optional
from textblob import TextBlob
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import uuid
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chatbot.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
load_dotenv()
app.config.update(
    SECRET_KEY=os.getenv("SECRET_KEY", os.urandom(24)),
    SESSION_TYPE='filesystem',
    SESSION_FILE_DIR=os.getenv("SESSION_FILE_DIR", str(Path.cwd() / "flask_session")),
    SESSION_PERMANENT=False,
    SESSION_FILE_THRESHOLD=500
)
Session(app)
Path(app.config['SESSION_FILE_DIR']).mkdir(exist_ok=True)

# Constants
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
INTENT_CLASSES = ["loan_application", "credit_card_application", "general_inquiry"]
MODEL_PATH = Path.cwd() / "fine_tuned_distilbert"

# Simulated training data
TRAINING_DATA = [
    ("I want a personal loan", "loan_application"),
    ("Can I get a mortgage?", "loan_application"),
    ("Apply for a Visa card", "credit_card_application"),
    ("Tell me about ATB C Jeune", "credit_card_application"),
    ("What are your savings accounts?", "general_inquiry"),
    ("How does ATB NET work?", "general_inquiry"),
]

# Load or fine-tune DistilBERT model
class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = [INTENT_CLASSES.index(label) for label in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_or_train_model():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    if MODEL_PATH.exists():
        logger.info("Loading fine-tuned model")
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    else:
        logger.info("Fine-tuning DistilBERT model")
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=len(INTENT_CLASSES)
        )
        texts = [x[0] for x in TRAINING_DATA]
        labels = [x[1] for x in TRAINING_DATA]
        dataset = IntentDataset(texts, labels, tokenizer)

        training_args = TrainingArguments(
            output_dir=str(MODEL_PATH / 'results'),
            num_train_epochs=3,
            per_device_train_batch_size=4,
            warmup_steps=10,
            weight_decay=0.01,
            logging_dir=str(MODEL_PATH / 'logs'),
            logging_steps=10,
            save_strategy="epoch",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()
        model.save_pretrained(MODEL_PATH)
        tokenizer.save_pretrained(MODEL_PATH)
    return model, tokenizer

model, tokenizer = load_or_train_model()
model.eval()

def predict_intent(text: str) -> str:
    """Predict intent using fine-tuned DistilBERT."""
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return INTENT_CLASSES[predicted_class]

def validate_user_input(user_input: str) -> bool:
    """Validate user input to prevent malicious content."""
    if not user_input or len(user_input) > 500:
        return False
    dangerous_patterns = [r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', r'javascript\s*:', r'on\w+\s*=']
    return not any(re.search(pattern, user_input, re.IGNORECASE) for pattern in dangerous_patterns)

def query_ollama(prompt: str) -> str:
    """Query Mistral LLM via Ollama API."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={'model': 'mistral', 'prompt': prompt, 'stream': False},
            timeout=30
        )
        response.raise_for_status()
        return response.json().get('response', 'Error: No response from LLM')
    except requests.RequestException as e:
        logger.error(f"Ollama API error: {str(e)}")
        return "Error: Failed to connect to LLM service"

def analyze_sentiment(text: str) -> Dict[str, float]:
    """Analyze sentiment of the input text using TextBlob."""
    blob = TextBlob(text)
    return {"polarity": blob.sentiment.polarity, "subjectivity": blob.sentiment.subjectivity}

def generate_session_summary(conversation: list) -> str:
    """Generate a summary of the session using LLM."""
    summary_prompt = f"""
    Summarize the following conversation in English, focusing on key user intents and outcomes:
    {conversation}
    """
    return query_ollama(summary_prompt)

def call_banking_api(action: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate banking API call."""
    logger.info(f"Calling banking API: action={action}")
    return {"status": "eligible", "service": "loan_department", "contact": "loan_officer@atb.tn"} if action == "check_eligibility" else {}

def generate_document(user_data: Dict[str, Any], doc_type: str) -> Dict[str, Any]:
    """Generate document based on user data and document type."""
    logger.info(f"Generating document: type={doc_type}")
    return {
        "document_type": doc_type,
        "name": user_data.get("name", "N/A"),
        "income": user_data.get("income", 0),
        "amount": user_data.get("amount", 0) if doc_type == "loan_application" else 0,
        "purpose": user_data.get("purpose", "N/A") if doc_type == "loan_application" else "N/A",
        "card_type": user_data.get("card_type", "N/A") if doc_type == "credit_card_application" else "N/A",
        "date": "2025-07-31",
        "status": "pending_confirmation"
    }

def generate_pdf(user_data: Dict[str, Any], doc_type: str) -> Optional[Path]:
    """Generate PDF form based on user data and document type."""
    filename = f"application_form_{uuid.uuid4().hex}.pdf"
    output_path = Path.cwd() / filename
    logger.info(f"Generating PDF: type={doc_type}, path={filename}")
    try:
        doc = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=inch, leftMargin=inch, topMargin=inch, bottomMargin=inch)
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        heading_style = styles['Heading2']
        normal_style = ParagraphStyle(name='Normal', fontSize=12, leading=14)
        content = [
            Paragraph("ATB Bank Application Form", title_style),
            Spacer(1, 0.2 * inch),
            Paragraph(f"Document Type: {doc_type.replace('_', ' ').title()}", heading_style),
            Spacer(1, 0.2 * inch),
            Paragraph("Applicant Information", heading_style),
            Paragraph(f"Name: {user_data.get('name', 'N/A')}", normal_style),
            Paragraph(f"Income: {user_data.get('income', 0)} TND", normal_style),
            Paragraph(f"Application Date: {user_data.get('date', '2025-07-31')}", normal_style),
            Paragraph(f"Status: {user_data.get('status', 'Pending Confirmation')}", normal_style),
            Spacer(1, 0.2 * inch),
            Paragraph("Application Details", heading_style),
        ]
        if doc_type == "loan_application":
            content.extend([
                Paragraph(f"Loan Amount: {user_data.get('amount', 0)} TND", normal_style),
                Paragraph(f"Purpose: {user_data.get('purpose', 'N/A')}", normal_style),
            ])
        else:
            content.append(Paragraph(f"Card Type: {user_data.get('card_type', 'N/A')}", normal_style))
        content.extend([
            Spacer(1, 0.2 * inch),
            Paragraph("Declaration", heading_style),
            Paragraph(f"I, {user_data.get('name', 'N/A')}, hereby declare that the information provided is accurate.", normal_style),
            Paragraph(f"Signature: ____________________ Date: {user_data.get('date', '2025-07-31')}", normal_style),
        ])
        doc.build(content)
        return output_path
    except Exception as e:
        logger.error(f"PDF generation error: {str(e)}")
        return None

@app.route('/')
def index():
    """Render the main index page."""
    logger.info("Rendering index page")
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests and manage form filling."""
    try:
        data = request.get_json()
        if not data:
            logger.error("No JSON data provided")
            return jsonify({"response": "No data provided"}), 400

        user_input = data.get('input', '').strip()[:500]
        language = data.get('language', 'en')
        user_data = data.get('user_data', {})

        if not validate_user_input(user_input):
            logger.warning("Invalid user input detected")
            return jsonify({"response": "Invalid input provided."}), 400

        # Initialize session if not present
        if 'form_state' not in session:
            session['form_state'] = {'step': 'initial', 'doc_type': None, 'user_data': {}, 'conversation': []}
            logger.info("Initialized new session")

        # Append user input to conversation history
        session['form_state']['conversation'].append(f"User: {user_input}")
        session.modified = True

        # Sentiment analysis
        sentiment = analyze_sentiment(user_input)
        logger.info(f"Sentiment analysis: polarity={sentiment['polarity']}, subjectivity={sentiment['subjectivity']}")

        if user_data:
            session['form_state']['user_data'].update(user_data)
            session.modified = True
            logger.info("Updated user data in session")

        form_state = session['form_state']
        output = {"response": "", "sentiment": sentiment}

        # Intent detection using DistilBERT
        if not form_state['doc_type']:
            predicted_intent = predict_intent(user_input)
            form_state['doc_type'] = predicted_intent if predicted_intent in ["loan_application", "credit_card_application"] else None
            if form_state['doc_type'] == "loan_application":
                form_state['required_fields'] = ["name", "income", "amount", "purpose"]
                logger.info("Detected loan intent")
            elif form_state['doc_type'] == "credit_card_application":
                form_state['required_fields'] = ["name", "income", "card_type"]
                logger.info("Detected credit card intent")

        if form_state['doc_type']:
            missing_fields = [field for field in form_state['required_fields'] if field not in form_state['user_data'] or not form_state['user_data'].get(field)]
            if missing_fields:
                field_prompts = {
                    "name": "Please provide your full name.",
                    "income": "Please indicate your monthly income (in TND).",
                    "amount": "Please specify the desired loan amount (in TND).",
                    "purpose": "Please state the purpose of the loan.",
                    "card_type": "Please specify the card type (e.g., Visa, Lella)."
                }
                output["response"] = field_prompts[missing_fields[0]]
                output["field_requested"] = missing_fields[0]
                session['form_state'] = form_state
                session.modified = True
                logger.info(f"Prompting for missing field: {missing_fields[0]}")
                return jsonify(output)

            document = generate_document(form_state['user_data'], form_state['doc_type'])
            output["document"] = document
            pdf_path = generate_pdf(form_state['user_data'], form_state['doc_type'])
            output["response"] = "Form generated successfully. Download the PDF below."
            output["pdf_available"] = bool(pdf_path)

            # Generate session summary
            summary = generate_session_summary(form_state['conversation'])
            output["session_summary"] = summary
            session['form_state']['conversation'].append(f"System: {output['response']}\nSummary: {summary}")
            session.pop('form_state', None)
            session.modified = True
            logger.info("PDF generated, session reset")
            return jsonify(output)

        prompt = f"""
        Banking assistant for ATB Bank (Tunisian bank). Respond in English.
        Input: "{user_input}".
        ATB Services: Savings/current accounts, cards (Lella, ATB C Jeune, Visa/Mastercard), loans (100 Jours SAKAN, Pack Intelligencia),
        ATB NET/Mobile, ATB Challenge, transfers, FATCA.
        Instructions: Provide information about ATB services or initiate data collection for loan/card applications.
        If the input doesn't match any service, respond: "Sorry, I didn't understand. Could you clarify your request?"
        User data: {form_state['user_data']}.
        """
        output["response"] = query_ollama(prompt)
        if "check eligibility" in user_input.lower():
            output["action"] = call_banking_api("check_eligibility", {})
        session['form_state']['conversation'].append(f"System: {output['response']}")
        session.modified = True
        return jsonify(output)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"response": f"Server error: {str(e)}"}), 500

@app.route('/download_pdf')
def download_pdf():
    """Serve the generated PDF file."""
    filename = f"application_form_{uuid.uuid4().hex}.pdf"
    pdf_path = Path.cwd() / filename
    if pdf_path.exists():
        logger.info(f"Serving PDF: {filename}")
        return send_file(pdf_path, as_attachment=True, download_name=filename)
    logger.error("PDF file not found")
    return jsonify({"response": "PDF file not available."}), 404

if __name__ == "__main__":
    logger.info("Starting Flask application")
    app.run(host='0.0.0.0', port=5000)