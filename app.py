import streamlit as st
import openai
import time
from datetime import datetime
import os
import tempfile
import hashlib
import json
from pathlib import Path

# Additional imports for RAG - CORRECTED IMPORTS
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader  # CORRECTED LINE
from langchain.embeddings.openai import OpenAIEmbeddings  # CORRECTED
from langchain.vectorstores import FAISS  # CORRECTED
from langchain.schema import Document

# Page configuration
st.set_page_config(
    page_title="Ask upSkill Air",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for RAG
if 'rag_initialized' not in st.session_state:
    st.session_state.rag_initialized = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = []
if 'processing_logs' not in st.session_state:
    st.session_state.processing_logs = []
if 'chunking_stats' not in st.session_state:
    st.session_state.chunking_stats = {}

def log_event(event_type, message, details=None):
    """Comprehensive logging function"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": event_type,
        "message": message,
        "details": details
    }
    st.session_state.processing_logs.append(log_entry)
    return log_entry

def validate_api_key(api_key):
    """Validate the OpenAI API key"""
    if not api_key:
        return False, "API key cannot be empty"
    
    if not api_key.startswith('sk-'):
        return False, "Invalid API key format. Should start with 'sk-'"
    
    return True, "API key format is valid"

def test_connection(api_key):
    """Test connection to OpenAI API"""
    try:
        client = openai.OpenAI(api_key=api_key)
        models = client.models.list()
        return True, "Connection successful!", len(models.data)
    except openai.AuthenticationError:
        return False, "Authentication failed: Invalid API key", 0
    except openai.APIConnectionError:
        return False, "Connection error: Unable to reach OpenAI servers", 0
    except Exception as e:
        return False, f"Error: {str(e)}", 0

def calculate_file_hash(file_content):
    """Calculate MD5 hash of file for duplicate detection"""
    hash_md5 = hashlib.md5()
    hash_md5.update(file_content)
    return hash_md5.hexdigest()

def count_tokens(text):
    """Count tokens in text using tiktoken"""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def load_document(file_path, file_extension):
    """Load document based on file type"""
    try:
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_extension in ['.docx', '.doc']:
            loader = Docx2txtLoader(file_path)
        else:
            # For unsupported types, use text loader as fallback
            loader = TextLoader(file_path, encoding='utf-8')
        
        documents = loader.load()
        log_event("FILE_LOADED", f"Successfully loaded {len(documents)} pages from document")
        return documents
    except Exception as e:
        log_event("FILE_ERROR", f"Error loading document", {"error": str(e)})
        return None

def process_documents(api_key, uploaded_files):
    """Process uploaded documents for RAG"""
    if not api_key:
        log_event("PROCESSING_ERROR", "No API key provided")
        return False
    
    all_documents = []
    processed_files = []
    
    log_event("PROCESSING_START", f"Starting processing of {len(uploaded_files)} files")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Check for duplicates
            file_hash = calculate_file_hash(uploaded_file.getvalue())
            if any(doc.get('hash') == file_hash for doc in st.session_state.documents_processed):
                log_event("DUPLICATE_SKIPPED", f"Skipped duplicate file: {uploaded_file.name}")
                continue
            
            status_text.text(f"Processing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
            
            # Save file temporarily
            file_extension = Path(uploaded_file.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Load document
            documents = load_document(tmp_file_path, file_extension)
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            if not documents:
                continue
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "file_name": uploaded_file.name,
                    "file_hash": file_hash,
                    "file_size": uploaded_file.size,
                    "upload_time": datetime.now().isoformat(),
                    "processed_by_api": api_key[:10] + "..."  # Track which API key processed this
                })
            
            all_documents.extend(documents)
            
            # Record processed document
            processed_file_info = {
                "name": uploaded_file.name,
                "hash": file_hash,
                "size": uploaded_file.size,
                "pages": len(documents),
                "processed_at": datetime.now().isoformat(),
                "api_key_owner": api_key[:10] + "..."  # Restrict to this API key
            }
            st.session_state.documents_processed.append(processed_file_info)
            processed_files.append(processed_file_info)
            
            log_event("FILE_PROCESSED", f"Processed {uploaded_file.name}", {
                "pages": len(documents),
                "size": uploaded_file.size
            })
            
        except Exception as e:
            log_event("PROCESSING_ERROR", f"Error processing {uploaded_file.name}", {"error": str(e)})
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    if not all_documents:
        log_event("PROCESSING_ERROR", "No documents were successfully processed")
        return False
    
    # Chunk documents
    status_text.text("Chunking documents...")
    log_event("CHUNKING_START", "Starting document chunking")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=count_tokens
    )
    
    chunks = text_splitter.split_documents(all_documents)
    
    # Calculate chunking statistics
    total_chunks = len(chunks)
    avg_chunk_size = sum(count_tokens(chunk.page_content) for chunk in chunks) / total_chunks if total_chunks > 0 else 0
    
    st.session_state.chunking_stats = {
        "total_documents": len(all_documents),
        "total_chunks": total_chunks,
        "avg_chunk_size": avg_chunk_size,
        "chunking_time": datetime.now().isoformat(),
        "processed_files": processed_files
    }
    
    log_event("CHUNKING_COMPLETE", "Document chunking completed", st.session_state.chunking_stats)
    
    # Create vector store
    status_text.text("Creating vector embeddings...")
    log_event("EMBEDDING_START", "Starting vector embedding creation")
    
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vector_store = FAISS.from_documents(chunks, embeddings)
        st.session_state.vector_store = vector_store
        st.session_state.rag_initialized = True
        
        log_event("RAG_INITIALIZED", "RAG system successfully initialized", {
            "total_chunks": total_chunks,
            "documents_processed": len(processed_files),
            "api_key_restricted": True
        })
        
        status_text.text("RAG system ready!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        return True
        
    except Exception as e:
        log_event("EMBEDDING_ERROR", "Error creating embeddings", {"error": str(e)})
        progress_bar.empty()
        status_text.empty()
        return False

def rag_query(question, api_key, max_results=3):
    """Perform RAG-based query"""
    if not st.session_state.vector_store:
        return None, []
    
    try:
        # Search for relevant documents
        relevant_docs = st.session_state.vector_store.similarity_search(
            question, 
            k=max_results
        )
        
        # Prepare context
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create enhanced prompt
        enhanced_prompt = f"""
        Based on the following context from uploaded documents, please answer the question. 
        If the context doesn't contain relevant information, indicate that clearly.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        
        # Get response from OpenAI
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Be precise and cite relevant information from the context when available."},
                {"role": "user", "content": enhanced_prompt}
            ],
            temperature=0.1
        )
        
        answer = response.choices[0].message.content
        
        log_event("RAG_QUERY", "RAG query processed", {
            "question": question,
            "documents_retrieved": len(relevant_docs),
            "answer_length": len(answer),
            "api_key_used": api_key[:10] + "..."
        })
        
        return answer, relevant_docs
        
    except Exception as e:
        log_event("RAG_ERROR", "Error processing RAG query", {"error": str(e)})
        return None, []

def main():
    st.title("upSkill Air Learning Assistant")
    st.markdown("---")
    
    # Sidebar for API configuration
    with st.sidebar:
        st.header("🔑 API Configuration")
        api_key = st.text_input(
            "Enter your OpenAI API Key:",
            type="password",
            placeholder="sk-...",
            help="Get your API key from https://platform.openai.com/api-keys"
        )
        
        if api_key:
            # Validate API key format
            is_valid, validation_msg = validate_api_key(api_key)
            if is_valid:
                st.success(validation_msg)
                
                # Test connection
                with st.spinner("Testing connection to OpenAI..."):
                    connection_success, connection_msg, model_count = test_connection(api_key)
                
                if connection_success:
                    st.success(connection_msg)
                    st.info(f"Available models: {model_count}")
                    
                    # Store API key in session state
                    st.session_state.api_key = api_key
                    st.session_state.connected = True
                else:
                    st.error(connection_msg)
                    st.session_state.connected = False
            else:
                st.error(validation_msg)
                st.session_state.connected = False
        else:
            st.warning("Please enter your OpenAI API key")
            st.session_state.connected = False
        
        st.markdown("---")
        
        # RAG Document Upload Section
        st.header("📁 RAG Document Setup")
        
        if st.session_state.get('connected', False):
            uploaded_files = st.file_uploader(
                "Choose documents for RAG",
                type=['pdf', 'txt', 'docx', 'doc'],
                accept_multiple_files=True,
                help="Upload PDF, TXT, or DOCX files to build your knowledge base"
            )
            
            if uploaded_files:
                if st.button("Process Documents for RAG"):
                    with st.spinner("Processing documents..."):
                        success = process_documents(api_key, uploaded_files)
                        if success:
                            st.success("✅ RAG system initialized successfully!")
                        else:
                            st.error("❌ Error initializing RAG system")
            
            # Display processed documents
            if st.session_state.documents_processed:
                st.subheader("Processed Documents")
                for doc in st.session_state.documents_processed:
                    st.write(f"📄 {doc['name']} ({doc['size']} bytes)")
            
            # Clear RAG data button
            if st.session_state.rag_initialized and st.button("Clear RAG Data"):
                st.session_state.rag_initialized = False
                st.session_state.vector_store = None
                st.session_state.documents_processed = []
                st.session_state.chunking_stats = {}
                log_event("RAG_CLEARED", "RAG data cleared by user")
                st.rerun()
                
        else:
            st.info("🔑 Connect to OpenAI first to enable RAG features")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Type your questions below")
        
        if st.session_state.get('connected', False):
            # Initialize chat history
            if 'messages' not in st.session_state:
                st.session_state.messages = []
            
            # Display RAG status
            if st.session_state.rag_initialized:
                st.success(f"✅ RAG Active - {len(st.session_state.documents_processed)} documents loaded")
            else:
                st.warning("⚠️ RAG Not Active - Upload documents to enable context-aware responses")
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message.get("sources"):
                        with st.expander("📎 View Sources"):
                            for i, source in enumerate(message["sources"]):
                                st.write(f"**Source {i+1}:** {source.metadata.get('file_name', 'Unknown')}")
                                st.write(f"**Content Preview:** {source.page_content[:200]}...")
            
            # Chat input
            if prompt := st.chat_input("Ask me anything..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate AI response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            if st.session_state.rag_initialized:
                                # Use RAG for response
                                answer, sources = rag_query(prompt, st.session_state.api_key)
                                if answer:
                                    st.markdown(answer)
                                    if sources:
                                        with st.expander("📎 View Sources"):
                                            for i, source in enumerate(sources):
                                                st.write(f"**Source {i+1}:** {source.metadata.get('file_name', 'Unknown')}")
                                                st.write(f"**Content Preview:** {source.page_content[:200]}...")
                                    
                                    # Add assistant response to chat history with sources
                                    st.session_state.messages.append({
                                        "role": "assistant", 
                                        "content": answer,
                                        "sources": sources
                                    })
                                else:
                                    st.error("Error generating RAG response")
                            else:
                                # Regular chat without RAG
                                client = openai.OpenAI(api_key=st.session_state.api_key)
                                
                                # Create completion with streaming
                                stream = client.chat.completions.create(
                                    model="gpt-3.5-turbo",
                                    messages=[
                                        {"role": m["role"], "content": m["content"]}
                                        for m in st.session_state.messages
                                    ],
                                    stream=True,
                                )
                                
                                response = st.write_stream(stream)
                                
                                # Add assistant response to chat history
                                st.session_state.messages.append({"role": "assistant", "content": response})
                                
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            # Clear chat button
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.rerun()
        
        else:
            st.info("🔑 Please enter a valid OpenAI API key in the sidebar to start chatting")
    
    with col2:
        st.header("📊 Telemetry")
        
        if st.session_state.get('connected', False):
            st.success("✅ Connected to OpenAI")
            st.metric("Connection Status", "Active")
            
            # RAG Status Section
            st.subheader("🦙 RAG Status")
            if st.session_state.rag_initialized:
                st.success("✅ RAG System Active", {len(st.session_state.documents_processed)} "documents loaded")
                st.metric("Documents Loaded", len(st.session_state.documents_processed))
                if st.session_state.chunking_stats:
                    st.metric("Total Chunks", st.session_state.chunking_stats.get('total_chunks', 0))
            else:
                st.warning("⚠️ RAG System Inactive")
            
            # Processing Statistics
            if st.session_state.chunking_stats:
                st.subheader("📈 Processing Statistics")
                stats = st.session_state.chunking_stats
                st.write(f"**Total Documents:** {stats.get('total_documents', 0)}")
                st.write(f"**Total Chunks:** {stats.get('total_chunks', 0)}")
                st.write(f"**Avg Chunk Size:** {stats.get('avg_chunk_size', 0):.1f} tokens")
                st.write(f"**Last Processed:** {stats.get('chunking_time', 'Never')}")
            
            # Display connection details
            st.subheader("🔗 Connection Details")
            st.write(f"**Last Test:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"**API Key:** `{st.session_state.api_key[:10]}...`")
            st.write("**Service:** OpenAI API")
            st.write("**Status:** Operational")
            
            # Chat statistics
            if st.session_state.messages:
                user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
                assistant_msgs = len([m for m in st.session_state.messages if m["role"] == "assistant"])
                
                st.subheader("💬 Chat Statistics")
                st.write(f"**User Messages:** {user_msgs}")
                st.write(f"**AI Responses:** {assistant_msgs}")
                st.write(f"**Total Messages:** {len(st.session_state.messages)}")
            
            # Processing Logs
            st.subheader("📋 Processing Logs")
            logs_to_show = st.session_state.processing_logs[-8:]  # Show last 8 logs
            for log in reversed(logs_to_show):
                log_time = datetime.fromisoformat(log['timestamp']).strftime('%H:%M:%S')
                icon = "❌" if "ERROR" in log['type'] else "✅" if "SUCCESS" in log['type'] else "ℹ️"
                st.write(f"{icon} **{log_time}** - {log['message']}")
                
            # Export logs button
            if st.session_state.processing_logs and st.button("Export Full Logs"):
                log_data = {
                    "export_time": datetime.now().isoformat(),
                    "total_logs": len(st.session_state.processing_logs),
                    "api_key_owner": st.session_state.api_key[:10] + "...",
                    "logs": st.session_state.processing_logs
                }
                st.download_button(
                    label="Download Logs JSON",
                    data=json.dumps(log_data, indent=2),
                    file_name=f"rag_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        else:
            st.warning("⏳ Waiting for connection...")

if __name__ == "__main__":
    main()
