import os
import time
import json
import openai
import hashlib
import tiktoken
import tempfile
import streamlit as st
from datetime import datetime
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from docx2txt import process as docx2txt_process
import psycopg2
from sqlalchemy import create_engine, text

# Default OpenAI API Key (used automatically if present)
# Priority: Streamlit secrets > Environment variable > Hardcoded default
try:
    DEFAULT_OPENAI_API_KEY = st.secrets.get("openai", {}).get("api_key", os.getenv("OPENAI_API_KEY", "sk-.."))
except:
    # Fallback if secrets.toml doesn't exist
    DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-..")

# PostgreSQL Configuration
# Priority: Streamlit secrets > Environment variables > Defaults
try:
    DEFAULT_PG_HOST = st.secrets.get("postgres", {}).get("host", os.getenv("POSTGRES_HOST", "localhost"))
    DEFAULT_PG_PORT = st.secrets.get("postgres", {}).get("port", os.getenv("POSTGRES_PORT", "5432"))
    DEFAULT_PG_DATABASE = st.secrets.get("postgres", {}).get("database", os.getenv("POSTGRES_DB", "upskill_rag"))
    DEFAULT_PG_USER = st.secrets.get("postgres", {}).get("user", os.getenv("POSTGRES_USER", "postgres"))
    DEFAULT_PG_PASSWORD = st.secrets.get("postgres", {}).get("password", os.getenv("POSTGRES_PASSWORD", ""))
except:
    # Fallback if secrets.toml doesn't exist
    DEFAULT_PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
    DEFAULT_PG_PORT = os.getenv("POSTGRES_PORT", "5432")
    DEFAULT_PG_DATABASE = os.getenv("POSTGRES_DB", "upskill_rag")
    DEFAULT_PG_USER = os.getenv("POSTGRES_USER", "postgres")
    DEFAULT_PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

# PGVector collection name
COLLECTION_NAME = "upskill_documents"

# Page configuration
st.set_page_config(
    page_title="Ask UpskillAir",
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

def get_pg_connection_string(host, port, database, user, password):
    """Create PostgreSQL connection string"""
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

def test_postgres_connection(connection_string):
    """Test PostgreSQL connection and initialize pgvector extension"""
    try:
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            # Check connection
            conn.execute(text("SELECT 1"))
            
            # Create pgvector extension if it doesn't exist
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            
        return True, "PostgreSQL connection successful and pgvector extension initialized"
    except Exception as e:
        return False, f"PostgreSQL connection failed: {str(e)}"

def get_vector_store(connection_string, api_key, create_new=False):
    """Get or create PGVector store"""
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        if create_new:
            # This will create the collection if it doesn't exist
            vector_store = PGVector(
                connection_string=connection_string,
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME,
            )
        else:
            # Connect to existing collection
            vector_store = PGVector(
                connection_string=connection_string,
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME,
            )
        
        return vector_store, None
    except Exception as e:
        return None, str(e)

def get_stored_documents(connection_string):
    """Retrieve list of stored documents from metadata"""
    try:
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            # Query the langchain_pg_embedding table for unique documents
            result = conn.execute(text(f"""
                SELECT DISTINCT 
                    cmetadata->>'file_name' as file_name,
                    cmetadata->>'file_hash' as file_hash,
                    cmetadata->>'file_size' as file_size,
                    cmetadata->>'upload_time' as upload_time
                FROM langchain_pg_embedding
                WHERE collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = :collection_name
                )
                AND cmetadata->>'file_name' IS NOT NULL
                ORDER BY cmetadata->>'upload_time' DESC
            """), {"collection_name": COLLECTION_NAME})
            
            documents = []
            for row in result:
                if row[0]:  # file_name exists
                    documents.append({
                        "name": row[0],
                        "hash": row[1],
                        "size": int(row[2]) if row[2] else 0,
                        "upload_time": row[3]
                    })
            
            return documents
    except Exception as e:
        log_event("DB_ERROR", "Error retrieving stored documents", {"error": str(e)})
        return []

def clear_vector_store(connection_string):
    """Clear all documents from the vector store"""
    try:
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            # Delete all embeddings for this collection
            conn.execute(text(f"""
                DELETE FROM langchain_pg_embedding
                WHERE collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = :collection_name
                )
            """), {"collection_name": COLLECTION_NAME})
            conn.commit()
        
        log_event("DB_CLEARED", "Vector store cleared successfully")
        return True
    except Exception as e:
        log_event("DB_ERROR", "Error clearing vector store", {"error": str(e)})
        return False

def load_document(file_path, file_extension):
    """Load document based on file type"""
    try:
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        elif file_extension == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
        elif file_extension in ['.docx', '.doc']:
            text_content = docx2txt_process(file_path)
            documents = [Document(page_content=text_content, metadata={"source": file_path})]
        else:
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
        
        log_event("FILE_LOADED", f"Successfully loaded {len(documents)} pages from document")
        return documents
    except Exception as e:
        log_event("FILE_ERROR", "Error loading document", {"error": str(e)})
        return None

def process_documents(api_key, uploaded_files, connection_string):
    """Process uploaded documents for RAG with PostgreSQL storage"""
    if not api_key:
        log_event("PROCESSING_ERROR", "No API key provided")
        return False
    
    if not connection_string:
        log_event("PROCESSING_ERROR", "No database connection string provided")
        return False
    
    # Get existing documents from database
    existing_docs = get_stored_documents(connection_string)
    existing_hashes = {doc['hash'] for doc in existing_docs}
    
    all_documents = []
    processed_files = []
    
    log_event("PROCESSING_START", f"Starting processing of {len(uploaded_files)} files")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Check for duplicates in database
            file_hash = calculate_file_hash(uploaded_file.getvalue())
            if file_hash in existing_hashes:
                log_event("DUPLICATE_SKIPPED", f"Skipped duplicate file (already in database): {uploaded_file.name}")
                status_text.text(f"Skipping {uploaded_file.name} (already in database)... ({i+1}/{len(uploaded_files)})")
                time.sleep(0.5)
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
                })
            
            all_documents.extend(documents)
            
            # Record processed document
            processed_file_info = {
                "name": uploaded_file.name,
                "hash": file_hash,
                "size": uploaded_file.size,
                "pages": len(documents),
                "processed_at": datetime.now().isoformat(),
            }
            processed_files.append(processed_file_info)
            
            log_event("FILE_PROCESSED", f"Processed {uploaded_file.name}", {
                "pages": len(documents),
                "size": uploaded_file.size
            })
            
        except Exception as e:
            log_event("PROCESSING_ERROR", f"Error processing {uploaded_file.name}", {"error": str(e)})
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    if not all_documents:
        log_event("PROCESSING_INFO", "No new documents to process")
        progress_bar.empty()
        status_text.empty()
        # Still initialize RAG with existing documents
        return init_rag_from_db(api_key, connection_string)
    
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
    
    log_event("CHUNKING_COMPLETE", "Document chunking completed", {
        "total_documents": len(all_documents),
        "total_chunks": total_chunks,
        "avg_chunk_size": avg_chunk_size
    })
    
    # Store in PostgreSQL vector store
    status_text.text("Storing vector embeddings in PostgreSQL...")
    log_event("EMBEDDING_START", "Starting vector embedding creation and storage")
    
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        # Get or create vector store
        if st.session_state.vector_store is None:
            # Create new vector store
            vector_store = PGVector.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                connection_string=connection_string,
            )
        else:
            # Add to existing vector store
            vector_store = st.session_state.vector_store
            vector_store.add_documents(chunks)
        
        st.session_state.vector_store = vector_store
        st.session_state.rag_initialized = True
        
        # Update documents processed list with all stored docs
        st.session_state.documents_processed = get_stored_documents(connection_string)
        
        log_event("RAG_INITIALIZED", "Documents stored in PostgreSQL successfully", {
            "new_chunks": total_chunks,
            "new_documents": len(processed_files),
            "total_documents_in_db": len(st.session_state.documents_processed)
        })
        
        status_text.text("Documents stored successfully!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        return True
        
    except Exception as e:
        log_event("EMBEDDING_ERROR", "Error creating/storing embeddings", {"error": str(e)})
        progress_bar.empty()
        status_text.empty()
        return False

def init_rag_from_db(api_key, connection_string):
    """Initialize RAG system from existing PostgreSQL data"""
    try:
        vector_store, error = get_vector_store(connection_string, api_key)
        if error:
            log_event("RAG_INIT_ERROR", "Error connecting to vector store", {"error": error})
            return False
        
        st.session_state.vector_store = vector_store
        st.session_state.rag_initialized = True
        
        # Load document metadata
        st.session_state.documents_processed = get_stored_documents(connection_string)
        
        log_event("RAG_INITIALIZED", "RAG system initialized from PostgreSQL", {
            "documents_in_db": len(st.session_state.documents_processed)
        })
        
        return True
    except Exception as e:
        log_event("RAG_INIT_ERROR", "Error initializing RAG from database", {"error": str(e)})
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
    st.title("UpskillAir Learning Assistant")
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔑 API Configuration")
        api_key = st.text_input(
            "Enter your OpenAI API Key:",
            type="password",
            placeholder="sk-...",
            value=DEFAULT_OPENAI_API_KEY,
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
        
        # PostgreSQL Configuration
        st.header("🗄️ PostgreSQL Configuration")
        
        with st.expander("Database Settings", expanded=False):
            pg_host = st.text_input("Host", value=DEFAULT_PG_HOST, help="PostgreSQL host address")
            pg_port = st.text_input("Port", value=DEFAULT_PG_PORT, help="PostgreSQL port")
            pg_database = st.text_input("Database", value=DEFAULT_PG_DATABASE, help="Database name")
            pg_user = st.text_input("User", value=DEFAULT_PG_USER, help="Database user")
            pg_password = st.text_input("Password", type="password", value=DEFAULT_PG_PASSWORD, help="Database password")
        
        # Create connection string
        connection_string = get_pg_connection_string(pg_host, pg_port, pg_database, pg_user, pg_password)
        
        # Test PostgreSQL connection
        if st.button("Test PostgreSQL Connection"):
            with st.spinner("Testing PostgreSQL connection..."):
                pg_success, pg_msg = test_postgres_connection(connection_string)
                if pg_success:
                    st.success(pg_msg)
                    st.session_state.pg_connected = True
                    st.session_state.connection_string = connection_string
                else:
                    st.error(pg_msg)
                    st.session_state.pg_connected = False
        
        # Show connection status
        if st.session_state.get('pg_connected', False):
            st.success("✅ PostgreSQL Connected")
            st.session_state.connection_string = connection_string
        else:
            st.warning("⚠️ PostgreSQL Not Connected")
        
        st.markdown("---")
        
        # RAG Document Upload Section
        st.header("📁 RAG Document Management")
        
        # Initialize RAG from database on first connection
        if (st.session_state.get('connected', False) and 
            st.session_state.get('pg_connected', False) and 
            not st.session_state.get('rag_initialized', False)):
            
            with st.spinner("Loading existing documents from database..."):
                if init_rag_from_db(api_key, connection_string):
                    if len(st.session_state.documents_processed) > 0:
                        st.success(f"✅ Loaded {len(st.session_state.documents_processed)} documents from database")
        
        if st.session_state.get('connected', False) and st.session_state.get('pg_connected', False):
            uploaded_files = st.file_uploader(
                "Upload New Documents",
                type=['pdf', 'txt', 'docx', 'doc'],
                accept_multiple_files=True,
                help="Upload PDF, TXT, or DOCX files. Duplicates will be automatically skipped."
            )
            
            if uploaded_files:
                if st.button("Process & Store Documents"):
                    with st.spinner("Processing documents..."):
                        success = process_documents(api_key, uploaded_files, connection_string)
                        if success:
                            st.success("✅ Documents stored in PostgreSQL successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Error processing documents")
            
            # Display stored documents from database
            if st.session_state.documents_processed:
                st.subheader(f"📚 Stored Documents ({len(st.session_state.documents_processed)})")
                for doc in st.session_state.documents_processed:
                    st.write(f"📄 {doc['name']} ({doc['size']} bytes)")
            
            # Clear database button
            if st.session_state.rag_initialized:
                st.markdown("---")
                if st.button("🗑️ Clear All Documents from Database", type="secondary"):
                    if clear_vector_store(connection_string):
                        st.session_state.rag_initialized = False
                        st.session_state.vector_store = None
                        st.session_state.documents_processed = []
                        st.session_state.chunking_stats = {}
                        st.success("✅ Database cleared successfully!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Error clearing database")
                
        else:
            if not st.session_state.get('connected', False):
                st.info("🔑 Connect to OpenAI first")
            if not st.session_state.get('pg_connected', False):
                st.info("🗄️ Connect to PostgreSQL to enable document storage")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Type your questions below")
        
        if st.session_state.get('connected', False):
            # Initialize chat history
            if 'messages' not in st.session_state:
                st.session_state.messages = []
            
            # Display RAG status
            #if st.session_state.rag_initialized:
               #st.success(".")
                # st.success(f"✅ RAG Active - {len(st.session_state.documents_processed)} documents loaded")
            #else:
               # st.warning("⚠️ RAG Not Active - Upload documents to enable context-aware responses")

            #if st.session_state.rag_initialized:
               #st.warning("⚠️ RAG Not Active - Upload documents to enable context-aware responses")
                #st.success(".")
                # st.success(f"✅ RAG Active - {len(st.session_state.documents_processed)} documents loaded")
            #else:
                
            
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
            st.metric("OpenAI Status", "Active")
            
            # PostgreSQL Status
            if st.session_state.get('pg_connected', False):
                st.success("✅ Connected to PostgreSQL")
                st.metric("PostgreSQL Status", "Active")
            else:
                st.warning("⚠️ PostgreSQL Disconnected")
            
            # RAG Status Section
            st.subheader("🦙 RAG Status")
            if st.session_state.rag_initialized:
                doc_count = len(st.session_state.documents_processed)
                st.success(f"✅ RAG System Active")
                st.metric("Documents in Database", doc_count)
                st.info("💾 Using PostgreSQL persistent storage")
            else:
                st.warning("⚠️ RAG System Inactive")
            
            # Display connection details
            st.subheader("🔗 Connection Details")
            st.write(f"**Last Check:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"**API Key:** `{st.session_state.api_key[:10]}...`")
            st.write("**OpenAI Service:** Operational")
            if st.session_state.get('pg_connected', False):
                st.write("**PostgreSQL:** Connected")
                st.write(f"**Database:** {DEFAULT_PG_DATABASE}")
            else:
                st.write("**PostgreSQL:** Not Connected")
            
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
