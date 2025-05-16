import streamlit as st
import os

def load_huggingface_token():
    """Load Hugging Face API token from Streamlit secrets or environment variables."""
    # Try to get token from Streamlit secrets
    if hasattr(st, "secrets") and "huggingface" in st.secrets and "api_token" in st.secrets["huggingface"]:
        return st.secrets["huggingface"]["api_token"]
    
    # Fallback to environment variable
    elif "HF_API_TOKEN" in os.environ:
        return os.environ["HF_API_TOKEN"]
    elif "HUGGINGFACE_API_TOKEN" in os.environ: # Common alternative
        return os.environ["HUGGINGFACE_API_TOKEN"]
    
    # No token found
    else:
        return None

def setup_api_access():
    """
    Set up API access for Hugging Face.
    If token is available, sets it as an environment variable.
    Also attempts to login to huggingface_hub if token is present.
    """
    token = load_huggingface_token()
    
    if token:
        # Set environment variable for Hugging Face libraries that might use it
        os.environ["HUGGINGFACE_API_TOKEN"] = token
        os.environ["HF_API_TOKEN"] = token # For compatibility with different libraries
        
        # For libraries that use transformers directly and might benefit from explicit login
        try:
            from huggingface_hub import login
            login(token=token)
            # st.sidebar.success("Successfully logged into Hugging Face Hub.") # Optional: can be verbose
        except ImportError:
            st.sidebar.warning("huggingface_hub library not found. Cannot perform explicit login. Ensure it is in requirements.txt if needed.")
        except Exception as e:
            st.sidebar.warning(f"Failed to login to Hugging Face Hub (token might still work for Inference API): {e}")
            # Continue execution as the token is still set in env vars and might be sufficient for HuggingFaceEndpoint
        return True
    else:
        st.sidebar.error("Hugging Face API token not found. Please ensure it is set in Streamlit secrets (secrets.toml) or as an environment variable (HUGGINGFACE_API_TOKEN or HF_API_TOKEN).")
        # st.stop() # Stopping might be too abrupt if the app has other functionalities or if token is optional for some parts.
        return False

# Example of how to use in your main app:
# if __name__ == "__main__":
#     if setup_api_access():
#         st.write("API access configured.")
#         # Proceed with app logic that requires the token
#     else:
#         st.error("API access could not be configured. Please check your token settings.")

