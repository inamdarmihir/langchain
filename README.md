Streamlit RAG Evaluation Application
This application allows you to upload a document, interact with it using a Retrieval Augmented Generation (RAG) pipeline, and evaluate the RAG system using RAGAS. It is built with Streamlit, LangChain, and RAGAS, utilizing open-source models like Mistral and Sentence Transformers.
Files Included
langchain_app.py: The main Streamlit application script.
secrets_handler.py: A utility script to handle Hugging Face API token access from Streamlit secrets or environment variables.
requirements.txt: A list of Python packages required to run the application.
Setup and Installation
Clone/Download Files:
Ensure you have langchain_app.py, secrets_handler.py, and requirements.txt in the same directory.
Create a Virtual Environment (Recommended):
bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install Dependencies:
Navigate to the directory containing the files and run:
bash
pip install -r requirements.txt
Set Up Hugging Face API Token:
This application requires a Hugging Face API token to access models (e.g., Mistral via HuggingFaceEndpoint) and potentially for RAGAS evaluation if using Hugging Face models for critique.
Create a directory named .streamlit in the same directory as your langchain_app.py file if it doesn_t already exist:
bash
mkdir .streamlit
Inside the .streamlit directory, create a file named secrets.toml.
Add your Hugging Face API token to secrets.toml in the following format:
toml
[huggingface]
api_token = "YOUR_HUGGINGFACE_API_TOKEN"
# Optional: If you are using a specific Hugging Face Inference Endpoint for your LLM
# HF_INFERENCE_ENDPOINT = "YOUR_MISTRAL_INFERENCE_ENDPOINT_URL"
Replace YOUR_HUGGINGFACE_API_TOKEN with your actual Hugging Face API token. If you don_t specify HF_INFERENCE_ENDPOINT, the application will default to a public Mistral-7B endpoint, which might have rate limits or availability issues. It_s recommended to use your own or a reliable endpoint.
Alternatively, you can set the HUGGINGFACE_API_TOKEN (or HF_API_TOKEN) as an environment variable.
Running the Application
Once the dependencies are installed and the API token is configured, run the Streamlit application from your terminal:
bash
streamlit run langchain_app.py
This will start the application, and your browser should open to the application_s local URL (usually http://localhost:8501 ).
Using the Application
Upload Document: Use the sidebar to upload a document (PDF, TXT, or CSV).
Chat with Document: Once the document is processed, you can ask questions in the chat interface. The RAG system will retrieve relevant information and generate answers.
RAGAS Evaluation Setup:
Specify the number of questions you want to use for evaluation.
For each question, provide the question itself and the corresponding ground truth answer.
Run RAGAS Evaluation: Click the button to run the evaluation. The application will use the RAGAS framework to assess metrics like faithfulness, answer relevancy, context precision, and context recall.
View Results: The evaluation results will be displayed in a table and visualized in a bar chart.
Notes
The application uses sentence-transformers/all-MiniLM-L6-v2 for embeddings and attempts to use a Mistral model via HuggingFaceEndpoint for generation.
Ensure your Hugging Face token has the necessary permissions if you are using gated models or private endpoints.
The performance and reliability of the LLM depend on the chosen Hugging Face Inference Endpoint.
