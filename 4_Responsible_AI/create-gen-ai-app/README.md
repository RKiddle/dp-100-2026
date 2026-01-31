Azure AI Foundry Chat Assistant
A Python-based generative AI application developed as part of the DP-100 (Designing and Implementing a Data Science Solution on Azure) certification prep. This project demonstrates how to connect a client application to an Azure AI Foundry project using the Microsoft Foundry Python SDK.

🚀 Overview
This application uses the gpt-4o model deployed via Azure AI Foundry to create a stateful chat experience. It showcases:

Authentication: Using DefaultAzureCredential for secure access.

SDK Integration: Utilizing azure-ai-projects and openai libraries.

Conversation Context: Managing chat history to allow for follow-up questions.

🛠️ Prerequisites
An active Azure Subscription.

An Azure AI Foundry Project with a gpt-4o model deployment.

Python 3.10 or higher.

⚙️ Setup & Configuration
Clone the repository:

Bash
git clone https://github.com/RKiddle/dp-100-2026.git
cd dp-100-2026
Install Dependencies:

Bash
pip install azure-identity azure-ai-projects openai python-dotenv
Environment Variables: Create a .env file in the root directory and add your specific Azure details:

Code snippet
PROJECT_ENDPOINT="your_foundry_project_endpoint"
MODEL_DEPLOYMENT="gpt-4o"
🖥️ Usage
Run the application using the following command:

Bash
python chat-app.py
Type your prompts into the terminal. Type quit to exit the session.

DP-100 Mastery Tip
In a real-world MLOps workflow, you would never check that .env file into GitHub. For the exam, remember that Azure Key Vault is the "gold standard" for storing secrets like your Project Endpoints and API keys.
