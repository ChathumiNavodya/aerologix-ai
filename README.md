<<<<<<< HEAD
# AeroLogix AI — Airport Operations Intelligence Dashboard

AeroLogix AI is an AI-powered airport operations dashboard that helps airport staff analyse flight delays, gate congestion, cargo and baggage activity, passenger impact, and operational recommendations.

The system combines data analytics, machine learning, an AI assistant, role-based access, PDF reporting, SQLite memory, and WhatsApp passenger alert support using Twilio Sandbox.

## Key Features

- Professional Streamlit dashboard with dark aerospace UI
- Admin and Passenger login roles
- Passenger-only flight status lookup
- CSV, XLSX, and XLS dataset upload
- Automated data cleaning and exploratory data analysis
- KPI dashboard for total flights, delay rate, average delay, cargo, and baggage
- Airline delay analysis
- Gate congestion and route analysis
- Cargo and baggage analytics
- AI-generated operational recommendations
- Natural-language AI Q&A assistant
- Optional LangGraph workflow for structured AI responses
- SQLite memory for sessions and chat history
- Random Forest ML delay prediction
- PDF report generation using ReportLab
- Twilio WhatsApp passenger alert support
- AI auto-alert trigger for high-delay flights

## Problem Statement

Airport operations involve flight schedules, gate allocation, cargo handling, baggage volume, passenger communication, and delay management. AeroLogix AI helps airport teams identify delay patterns, understand congestion points, predict delay risk, and generate passenger-friendly alerts from operational data.

## Dataset Columns

```text
Flight_ID, Airline, Origin, Destination, Departure_Time, Arrival_Time,
Delay_Minutes, Cargo_Weight, Baggage_Count, Gate_Number, Flight_Status
```

Sample dataset:

```text
data/sample_airport_data.csv
```

## Tech Stack

- Python
- Streamlit
- Pandas / NumPy
- Plotly / Matplotlib
- Scikit-learn
- LangChain
- LangGraph
- Groq API / Google Gemini API
- SQLite
- ReportLab
- Twilio WhatsApp Sandbox
- python-dotenv

## Project Structure

```text
aerologix-ai/
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── data/
│   └── sample_airport_data.csv
├── modules/
│   ├── __init__.py
│   ├── agent_graph.py
│   ├── api.py
│   ├── auth.py
│   ├── data_cleaner.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── llm_agent.py
│   ├── memory.py
│   ├── ml_predictor.py
│   ├── pdf_report.py
│   ├── recommender.py
│   ├── report_generator.py
│   ├── visualizer.py
│   └── whatsapp_service.py
├── reports/
└── screenshots/
```

## Installation

```bash
git clone https://github.com/ChathumiNavodya/aerologix-ai.git
cd aerologix-ai
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
LLM_PROVIDER=auto

TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_WHATSAPP_FROM=whatsapp:.............
```

Important:

```text
Do not upload .env to GitHub.
Use TWILIO_WHATSAPP_FROM=whatsapp:........... for Twilio Sandbox.
```

## Login Roles

### Admin

Admin users can access dashboard, analytics, ML predictor, reports, AI assistant, passenger alerts, and AI auto alerts.

### Passenger

Passenger users can access flight status lookup, passenger delay advice, and WhatsApp message preview.

## WhatsApp Passenger Alerts

This project supports passenger alert messages using Twilio WhatsApp Sandbox.

To test WhatsApp:

1. Open Twilio WhatsApp Sandbox.
2. Send the sandbox join code from your WhatsApp number.
3. Use this sender in `.env`:

```env
TWILIO_WHATSAPP_FROM=whatsapp:+94.......
```

4. Enter passenger number in the app using either format:

## AI Agent Components

- **Planning:** Generates analysis plans after dataset upload.
- **Reasoning:** Explains delay causes, congestion patterns, and passenger impact.
- **Memory:** Stores sessions, questions, answers, and insights in SQLite.
- **Interaction:** Provides Streamlit UI and natural-language AI Q&A.
- **Automation:** Cleans data, generates charts, creates recommendations, predicts delay risk, builds reports, and triggers alerts.

## Machine Learning

The ML module uses a Random Forest model to predict flight delay risk based on airline, origin, destination, gate number, departure hour, cargo weight, and baggage count.

## Deployment

Recommended platform: **Streamlit Community Cloud**

Steps:

1. Push this project to GitHub.
2. Go to Streamlit Cloud.
3. Create a new app.
4. Select the GitHub repository.
5. Set main file path as:

```text
app.py
```

6. Add API keys in Streamlit Secrets.

Example Streamlit Secrets:

```toml
GROQ_API_KEY = "your_groq_key"
GOOGLE_API_KEY = "your_google_key"
LLM_PROVIDER = "auto"

TWILIO_ACCOUNT_SID = "your_twilio_sid"
TWILIO_AUTH_TOKEN = "your_twilio_token"
TWILIO_WHATSAPP_FROM = "whatsapp:+94........"
```

## Security Notes

Do not commit these files to GitHub:

```text
.env
venv/
__pycache__/
*.pyc
*.db
reports/
screenshots/
```

## Future Improvements

- Live airport API integration
- Real passenger database with flight-linked phone numbers
- Advanced time-series delay forecasting
- Production WhatsApp Business API integration
- Cloud database deployment
- More advanced role-based permissions

## Author

Developed by Chathumi Navodya.
