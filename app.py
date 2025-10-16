import streamlit as st
import openai
import time
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="OpenAI Chat Interface",
    page_icon="🤖",
    layout="wide"
)

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

def main():
    st.title("🤖 OpenAI Chat Interface")
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
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        if st.session_state.get('connected', False):
            # Initialize chat history
            if 'messages' not in st.session_state:
                st.session_state.messages = []
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
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
            
            # Display connection details
            st.subheader("Connection Details")
            st.write(f"**Last Test:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"**API Key:** `{st.session_state.api_key[:10]}...`")
            st.write("**Service:** OpenAI API")
            st.write("**Status:** Operational")
            
            # Chat statistics
            if st.session_state.messages:
                user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
                assistant_msgs = len([m for m in st.session_state.messages if m["role"] == "assistant"])
                
                st.subheader("Chat Statistics")
                st.write(f"**User Messages:** {user_msgs}")
                st.write(f"**AI Responses:** {assistant_msgs}")
                st.write(f"**Total Messages:** {len(st.session_state.messages)}")
        else:
            st.warning("⏳ Waiting for connection...")

if __name__ == "__main__":
    main()
