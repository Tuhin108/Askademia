"""
Streamlit Web Interface for Document Q&A AI Agent
Enterprise-Ready Implementation with Security and Performance Features
"""

import streamlit as st
import os
import tempfile
from typing import Optional
import logging
from pathlib import Path

# Import our main agent
from main import DocumentQAAgent, logger

# Configure Streamlit page
st.set_page_config(
    page_title="Askademia",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    .document-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

def initialize_agent(api_key: str) -> bool:
    """Initialize the Document Q&A Agent"""
    try:
        st.session_state.agent = DocumentQAAgent(api_key)
        st.session_state.initialized = True
        return True
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        return False

def save_uploaded_file(uploaded_file) -> Optional[str]:
    """Save uploaded file to temporary directory"""
    try:
        # Create a temporary file
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def display_document_info(doc_data: dict, doc_id: str):
    """Display document information in a formatted card"""
    st.markdown(f"""
    <div class="document-card">
        <h4>📄 {doc_data.get('title', 'Untitled Document')}</h4>
        <p><strong>Document ID:</strong> {doc_id}</p>
        <p><strong>Processed:</strong> {doc_data.get('processed_at', 'N/A')}</p>
        <p><strong>Sections:</strong> {len(doc_data.get('sections', []))}</p>
        <p><strong>Tables:</strong> {len(doc_data.get('tables', []))}</p>
        <p><strong>Figures:</strong> {len(doc_data.get('figures', []))}</p>
        <p><strong>File:</strong> {os.path.basename(doc_data.get('file_path', ''))}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if doc_data.get('abstract'):
        with st.expander("View Abstract"):
            st.write(doc_data['abstract'])

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Askademia</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Advanced multi-modal document processing with ArXiv integration</p>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Enter your Gemini API key to initialize the agent"
        )
        
        if api_key and not st.session_state.initialized:
            if st.button("Initialize Agent", type="primary"):
                with st.spinner("Initializing AI Agent..."):
                    if initialize_agent(api_key):
                        st.success("✅ Agent initialized successfully!")
                        st.experimental_rerun()
        
        if st.session_state.initialized:
            st.success("✅ Agent Ready")
            
            st.markdown("---")
            st.markdown("### 📊 System Status")
            
            if st.session_state.agent:
                num_docs = len(st.session_state.agent.processed_documents)
                st.metric("Documents Processed", num_docs)
                st.metric("Queries Processed", len(st.session_state.query_history))
            
            st.markdown("---")
            st.markdown("### 🔧 Actions")
            
            if st.button("Clear All Documents"):
                if st.session_state.agent:
                    st.session_state.agent.processed_documents.clear()
                    st.success("All documents cleared!")
                    st.experimental_rerun()
            
            if st.button("Clear Query History"):
                st.session_state.query_history.clear()
                st.success("Query history cleared!")
                st.experimental_rerun()
    
    # Main content area
    if not st.session_state.initialized:
        st.markdown("""
        <div class="info-box">
            <h3>🚀 Welcome to askademia</h3>
            <p>This advanced AI agent provides:</p>
            <ul>
                <li><strong>Multi-modal PDF Processing:</strong> Extract text, tables, figures, and structure</li>
                <li><strong>Intelligent Q&A:</strong> Context-aware responses with query optimization</li>
                <li><strong>ArXiv Integration:</strong> Search and download academic papers</li>
                <li><strong>Enterprise Security:</strong> Input sanitization and rate limiting</li>
            </ul>
            <p><strong>To get started:</strong> Enter your Google Gemini API key in the sidebar and click "Initialize Agent".</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Document Upload", "💬 Q&A Interface", "🔍 ArXiv Search", "📊 Document Manager"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">📄 Document Upload & Processing</h2>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF documents for processing"
        )
        
        if uploaded_files:
            st.markdown("### Uploaded Files:")
            for file in uploaded_files:
                st.write(f"📄 {file.name} ({file.size} bytes)")
            
            if st.button("Process Documents", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Save file temporarily
                    file_path = save_uploaded_file(uploaded_file)
                    
                    if file_path:
                        try:
                            # Process document
                            doc_id = st.session_state.agent.ingest_document(file_path)
                            
                            st.markdown(f"""
                            <div class="success-box">
                                ✅ Successfully processed: {uploaded_file.name} (ID: {doc_id})
                            </div>
                            """, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.markdown(f"""
                            <div class="error-box">
                                ❌ Failed to process {uploaded_file.name}: {str(e)}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        finally:
                            # Clean up temporary file
                            try:
                                os.unlink(file_path)
                            except:
                                pass
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("✅ Processing complete!")
                st.experimental_rerun()
    
    with tab2:
        st.markdown('<h2 class="sub-header">💬 Intelligent Q&A Interface</h2>', unsafe_allow_html=True)
        
        if not st.session_state.agent.processed_documents:
            st.warning("⚠️ No documents available. Please upload PDF documents first.")
            return
        
        # Document selection
        doc_options = ["All Documents"] + list(st.session_state.agent.processed_documents.keys())
        doc_titles = ["All Documents"] + [
            f"{doc_id}: {data.get('title', 'Untitled')[:50]}..."
            for doc_id, data in st.session_state.agent.processed_documents.items()
        ]
        
        selected_doc_display = st.selectbox(
            "Select Document(s) to Query",
            options=doc_titles,
            help="Choose a specific document or query all documents"
        )

        if selected_doc_display and selected_doc_display in doc_titles:
            selected_doc = doc_options[doc_titles.index(selected_doc_display)]
            if selected_doc == "All Documents":
                selected_doc = None
        else:
            selected_doc = None
        
        # Query input with example suggestions
        st.markdown("### Query Examples:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎯 What is the main conclusion?"):
                st.session_state.current_query = "What is the main conclusion of this paper?"
        
        with col2:
            if st.button("📊 Show evaluation results"):
                st.session_state.current_query = "What are the accuracy and F1-score reported in this paper?"
        
        with col3:
            if st.button("🔬 Explain methodology"):
                st.session_state.current_query = "Summarize the methodology used in this paper."
        
        # Query input
        query = st.text_area(
            "Enter your question:",
            value=getattr(st.session_state, 'current_query', '') or '',
            height=100,
            help="Ask questions about the document content, methodology, results, or conclusions"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🚀 Ask Question", type="primary", disabled=not query.strip()):
                if query.strip():
                    with st.spinner("🤔 Analyzing document and generating response..."):
                        response = st.session_state.agent.query_document(query, selected_doc)
                        
                        # Store in history
                        st.session_state.query_history.append({
                            'query': query,
                            'response': response,
                            'document': selected_doc or "All Documents",
                            'timestamp': st.session_state.agent.security_manager.SecurityManager().rate_limit.__defaults__[0] if hasattr(st.session_state.agent.security_manager, 'SecurityManager') else "N/A"
                        })
                        
                        # Display response
                        st.markdown("### 🤖 AI Response:")
                        st.markdown(f"""
                        <div class="info-box">
                            {response}
                        </div>
                        """, unsafe_allow_html=True)
        
        with col2:
            if st.button("🗑️ Clear Query"):
                if hasattr(st.session_state, 'current_query'):
                    del st.session_state.current_query
                st.experimental_rerun()
        
        # Query History
        if st.session_state.query_history:
            st.markdown("---")
            st.markdown("### 📝 Recent Queries")
            
            for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
                with st.expander(f"Q: {item['query'][:100]}..." if len(item['query']) > 100 else f"Q: {item['query']}"):
                    st.markdown(f"**Document:** {item['document']}")
                    st.markdown(f"**Response:** {item['response']}")
    
    with tab3:
        st.markdown('<h2 class="sub-header">🔍 ArXiv Paper Search & Integration</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <p>Search for academic papers on ArXiv and automatically download and process them.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # ArXiv search
        search_query = st.text_input(
            "Search ArXiv Papers",
            placeholder="e.g., machine learning transformers, computer vision attention",
            help="Enter keywords to search for relevant papers on ArXiv"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            max_results = st.slider("Maximum Results", min_value=1, max_value=10, value=5)
        
        if st.button("🔍 Search ArXiv", type="primary", disabled=not search_query.strip()):
            if search_query.strip():
                with st.spinner("🔍 Searching ArXiv database..."):
                    try:
                        papers = st.session_state.agent.arxiv_integration.search_papers(search_query, max_results)
                        
                        if papers:
                            st.markdown(f"### 📚 Found {len(papers)} papers:")
                            
                            for i, paper in enumerate(papers):
                                with st.expander(f"📄 {paper['title']}"):
                                    col1, col2 = st.columns([3, 1])
                                    
                                    with col1:
                                        st.markdown(f"**Authors:** {', '.join(paper['authors'])}")
                                        st.markdown(f"**Published:** {paper['published']}")
                                        st.markdown(f"**Categories:** {', '.join(paper['categories'])}")
                                        st.markdown(f"**Summary:** {paper['summary'][:300]}...")
                                    
                                    with col2:
                                        st.markdown(f"[📖 View PDF]({paper['pdf_url']})")
                                        
                                        # Download & Process button removed as per user request
                                        # if st.button(f"⬇️ Download & Process", key=f"download_{i}"):
                                        #     pdf_url = paper.get('pdf_url')
                                        #     if not pdf_url:
                                        #         st.error(f"❌ No PDF URL available for this paper")
                                        #         continue
                                        #
                                        #     st.info(f"Debug: Attempting to download from URL: {pdf_url}")
                                        #     with st.spinner(f"Downloading and processing {paper['title'][:50]}..."):
                                        #         try:
                                        #             # Use the download_paper method directly for debugging
                                        #             file_path = st.session_state.agent.arxiv_integration.download_paper(pdf_url)
                                        #             if file_path:
                                        #                 doc_id = st.session_state.agent.ingest_document(file_path)
                                        #                 st.success(f"✅ Paper downloaded and processed! Document ID: {doc_id}")
                                        #                 st.experimental_rerun()
                                        #             else:
                                        #                 st.error("❌ Failed to download paper - check logs for details")
                                        #                 st.info("Check the ai_agent.log file for more details on the download failure.")
                                        #         except Exception as e:
                                        #             st.error(f"❌ Error: {str(e)}")
                                        #             logger.error(f"Download button error for paper {i}: {e}")
                        else:
                            st.warning("No papers found for your search query.")
                    
                    except Exception as e:
                        st.error(f"Search failed: {str(e)}")
    
    with tab4:
        st.markdown('<h2 class="sub-header">📊 Document Management</h2>', unsafe_allow_html=True)
        
        if not st.session_state.agent.processed_documents:
            st.info("📝 No documents have been processed yet.")
            return
        
        st.markdown(f"### 📁 Processed Documents ({len(st.session_state.agent.processed_documents)})")
        
        # Display all documents
        for doc_id, doc_data in st.session_state.agent.processed_documents.items():
            display_document_info(doc_data, doc_id)
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                if st.button(f"📄 View Summary", key=f"summary_{doc_id}"):
                    summary = st.session_state.agent.get_document_summary(doc_id)
                    st.markdown(f"""
                    <div class="info-box">
                        {summary}
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                if st.button(f"💬 Quick Query", key=f"query_{doc_id}"):
                    # Set up for quick query in Q&A tab
                    st.session_state.selected_doc_for_query = doc_id
                    st.info(f"Switch to Q&A tab to query document {doc_id}")
            
            with col3:
                if st.button(f"🗑️ Remove", key=f"remove_{doc_id}"):
                    del st.session_state.agent.processed_documents[doc_id]
                    st.success(f"Document {doc_id} removed!")
                    st.experimental_rerun()
            
            st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🤖 <strong>askademia</strong> | made by Tuhin ❤️</p>
        <p>Features: Multi-modal Processing • Context Optimization • Enterprise Security • ArXiv Integration</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()