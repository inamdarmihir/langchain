import streamlit as st
import os
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
from typing import List, Dict, Any, Optional

# Import secrets handler
from secrets_handler import setup_api_access

# Langchain imports
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import HuggingFaceEndpoint

# RAGAS evaluation imports
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall,
    context_precision,
)
from ragas.metrics.critique import harmfulness
from ragas.llm import HuggingFaceEvaluationLLM
from ragas import evaluate

# Set page configuration
st.set_page_config(
    page_title="RAG Evaluation with RAGAS & LangChain",
    page_icon="🧪",
    layout="wide"
)

# Initialize session states
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None
if "documents" not in st.session_state:
    st.session_state.documents = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "eval_data" not in st.session_state:
    st.session_state.eval_data = None
if "history" not in st.session_state:
    st.session_state.history = []

def initialize_llm():
    """Initialize the open-source LLM (Mistral)."""
    try:
        # Ensure HF token is available
        token = os.environ.get("HUGGINGFACE_API_TOKEN") or os.environ.get("HF_API_TOKEN")
        if not token:
            st.error("Hugging Face API token not found. Please set up your token first.")
            st.stop()

        # Initialize HuggingFaceEndpoint for Mistral-7B or other suitable model
        # Ensure the endpoint_url points to a valid Hugging Face Inference API endpoint for your chosen model
        # For example, for Mistral-7B-Instruct-v0.1: "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
        # You might need to adjust parameters like max_new_tokens based on the model and your needs.
        llm = HuggingFaceEndpoint(
            endpoint_url=st.secrets.get("HF_INFERENCE_ENDPOINT", "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"),
            huggingfacehub_api_token=token,
            task="text-generation",
            model_kwargs={
                "max_new_tokens": 512,
                "top_k": 50,
                "temperature": 0.1, # Low temperature for more factual RAG
                "repetition_penalty": 1.03,
            }
        )
        st.session_state.llm = llm
        # Also set up the RAGAS LLM wrapper
        st.session_state.ragas_llm = HuggingFaceEvaluationLLM(llm=llm)
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        st.stop()

def initialize_embeddings():
    """Initialize HuggingFace embeddings."""
    try:
        # Using a common, effective sentence transformer model
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"} # Explicitly use CPU for wider compatibility
        )
        st.session_state.embeddings = embeddings
        return embeddings
    except Exception as e:
        st.error(f"Error initializing embeddings: {e}")
        st.stop()

# Load and process document
def load_and_process_document(uploaded_file):
    if uploaded_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            # Determine loader based on file type
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_file_path)
            elif uploaded_file.name.endswith(".txt"):
                loader = TextLoader(tmp_file_path)
            elif uploaded_file.name.endswith(".csv"):
                loader = CSVLoader(tmp_file_path)
            else:
                st.error("Unsupported file type. Please upload PDF, TXT, or CSV.")
                return None, None
            
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)
            
            os.remove(tmp_file_path) # Clean up temp file
            return documents, texts
        except Exception as e:
            st.error(f"Error loading or processing document: {e}")
            return None, None
    return None, None

# Create vector store
def create_vector_store(texts, embeddings):
    if texts and embeddings:
        try:
            vectorstore = FAISS.from_documents(texts, embeddings)
            return vectorstore
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            return None
    return None

# Build RAG chain
def build_rag_chain(vectorstore, llm):
    if vectorstore and llm:
        retriever = vectorstore.as_retriever()
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | llm 
            | StrOutputParser()
        )
        return rag_chain
    return None

# Prepare data for RAGAS evaluation
def prepare_eval_data(questions: List[str], rag_chain, ground_truths: List[List[str]]):
    answers = []
    contexts = []
    if not rag_chain:
        st.error("RAG chain not initialized. Cannot prepare evaluation data.")
        return None

    for question in questions:
        try:
            # Assuming the RAG chain directly gives the answer
            # To get contexts, we need to access the retriever part separately or modify the chain output
            # For simplicity, let's assume the chain is modified or we can tap into the retriever
            # This part might need adjustment based on how `build_rag_chain` is structured for context retrieval
            
            # Invoke the chain for the answer
            response = rag_chain.invoke(question)
            answers.append(response)
            
            # Retrieve contexts separately for RAGAS
            # This is a common pattern: get contexts that led to the answer
            retriever = st.session_state.vectorstore.as_retriever()
            retrieved_docs = retriever.get_relevant_documents(question)
            contexts.append([doc.page_content for doc in retrieved_docs])
            
        except Exception as e:
            st.warning(f"Error processing question \n`{question}`\n for RAGAS: {e}")
            answers.append("Error generating answer.")
            contexts.append([]) # Empty context if error

    if len(questions) != len(ground_truths):
        st.error("Number of questions and ground truths must match for RAGAS evaluation.")
        return None

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths # RAGAS expects this column name
    }
    return data

# Run RAGAS evaluation
def run_ragas_evaluation(eval_data, ragas_llm, embeddings):
    if eval_data and ragas_llm and embeddings:
        from datasets import Dataset
        dataset = Dataset.from_dict(eval_data)
        
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            # harmfulness # Uncomment if critique is needed and configured
        ]
        
        try:
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=ragas_llm, # Using the HuggingFaceEvaluationLLM wrapper
                embeddings=embeddings # Using LangchainEmbeddingsWrapper for HF embeddings
            )
            return result
        except Exception as e:
            st.error(f"Error during RAGAS evaluation: {e}")
            return None
    return None

# --- Streamlit App UI ---
st.title("🧪 RAG Evaluation System with RAGAS & LangChain")
st.markdown("""
Welcome! This application allows you to upload a document, ask questions against it using a RAG (Retrieval Augmented Generation) 
            pipeline powered by open-source models (Mistral & Sentence Transformers), and then evaluate the RAG system using RAGAS.
""")

# Initialize API access (checks for secrets)
setup_api_access()

# Initialize LLM and Embeddings once
if "llm" not in st.session_state:
    st.session_state.llm = initialize_llm()
if "embeddings" not in st.session_state:
    st.session_state.embeddings = initialize_embeddings()

llm = st.session_state.llm
embeddings = st.session_state.embeddings
ragas_llm = st.session_state.get("ragas_llm") # Should be set during initialize_llm

# --- Sidebar for Configuration ---
st.sidebar.header("⚙️ Configuration")

# Document Upload
st.sidebar.subheader("1. Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload your document (PDF, TXT, CSV)", type=["pdf", "txt", "csv"])

if uploaded_file:
    if st.session_state.get("uploaded_file_name") != uploaded_file.name:
        with st.spinner("Processing document..."):
            docs, texts = load_and_process_document(uploaded_file)
            if docs and texts:
                st.session_state.documents = docs
                st.session_state.vectorstore = create_vector_store(texts, embeddings)
                st.session_state.uploaded_file_name = uploaded_file.name
                st.sidebar.success(f"Document `{uploaded_file.name}` processed and vector store created!")
            else:
                st.sidebar.error("Failed to process document.")
                st.session_state.documents = None
                st.session_state.vectorstore = None
                st.session_state.uploaded_file_name = None
elif st.session_state.get("uploaded_file_name"):
    st.sidebar.info(f"Using previously uploaded document: 
`{st.session_state.uploaded_file_name}`
")

# --- Main Area for Interaction and Evaluation ---
if st.session_state.vectorstore and llm:
    st.header("💬 Chat with your Document (RAG)")
    rag_chain = build_rag_chain(st.session_state.vectorstore, llm)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if rag_chain:
                    response = rag_chain.invoke(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    # Add to history for RAGAS
                    st.session_state.history.append({"question": prompt, "answer": response})
                else:
                    st.error("RAG chain not available.")
    
    st.markdown("---    ")
    st.header("📊 RAGAS Evaluation Setup")

    num_eval_questions = st.number_input("Number of questions for RAGAS evaluation", min_value=1, max_value=10, value=3, step=1)
    
    eval_questions_list = []
    eval_ground_truths_list = []

    st.markdown("Enter your evaluation questions and their corresponding ground truth answers:")
    for i in range(num_eval_questions):
        cols = st.columns([1, 2])
        with cols[0]:
            q = st.text_input(f"Question {i+1}", key=f"q_{i}", placeholder="E.g., What is the main topic?")
        with cols[1]:
            gt = st.text_area(f"Ground Truth {i+1}", key=f"gt_{i}", placeholder="E.g., The main topic is about renewable energy.")
        
        if q and gt:
            eval_questions_list.append(q)
            eval_ground_truths_list.append([gt]) # RAGAS expects list of strings for ground_truth

    if st.button("🚀 Run RAGAS Evaluation", disabled=not (eval_questions_list and len(eval_questions_list) == num_eval_questions)):
        if not ragas_llm:
            st.error("RAGAS LLM not initialized. Cannot run evaluation.")
        elif not embeddings:
            st.error("Embeddings not initialized. Cannot run evaluation.")
        else:
            with st.spinner("Preparing evaluation data and running RAGAS..."):
                # Use the RAG chain to get answers and contexts for the eval questions
                st.session_state.eval_data = prepare_eval_data(eval_questions_list, rag_chain, eval_ground_truths_list)
                
                if st.session_state.eval_data:
                    st.session_state.evaluation_results = run_ragas_evaluation(
                        st.session_state.eval_data, 
                        ragas_llm, 
                        st.session_state.embeddings
                    )
                    if st.session_state.evaluation_results:
                        st.success("RAGAS Evaluation Completed!")
                    else:
                        st.error("RAGAS Evaluation failed or returned no results.")
                else:
                    st.error("Failed to prepare data for RAGAS evaluation.")

    # Display RAGAS Results
    if st.session_state.evaluation_results:
        st.subheader("📈 Evaluation Results")
        results_df = st.session_state.evaluation_results.to_pandas()
        st.dataframe(results_df)

        # Visualizations (Example: Bar chart for scores)
        st.markdown("#### Metric Scores Overview")
        # Filter out non-numeric columns or select specific metrics for plotting
        metrics_for_plot = [col for col in results_df.columns if isinstance(results_df[col].iloc[0], (int, float))]
        if metrics_for_plot:
            avg_scores = results_df[metrics_for_plot].mean().reset_index()
            avg_scores.columns = ["metric", "average_score"]
            fig = px.bar(avg_scores, x="metric", y="average_score", title="Average RAGAS Metric Scores", color="metric")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric metrics found to plot.")

        # Display detailed results per question if needed
        st.markdown("#### Detailed Results per Question")
        for i, row in results_df.iterrows():
            with st.expander(f"Question: {row["question"]}"):
                st.write(f"**Answer:** {row["answer"]}")
                st.write(f"**Contexts Provided:**")
                for idx, ctx in enumerate(row["contexts"]):
                    st.text_area(f"Context {idx+1}", ctx, height=100, disabled=True, key=f"ctx_detail_{i}_{idx}")
                st.write("**Metrics:**")
                # Display individual metric scores for this question
                metric_scores_question = {m: row[m] for m in metrics_for_plot if m in row}
                st.json(metric_scores_question)

else:
    if not uploaded_file:
        st.info("Please upload a document in the sidebar to begin.")
    elif not llm:
        st.error("LLM not initialized. Check API token and configuration.")
    elif not embeddings:
        st.error("Embeddings not initialized.")

st.sidebar.markdown("---    ")
st.sidebar.markdown("**About:** This app demonstrates RAG evaluation using LangChain and RAGAS with open-source models.")

