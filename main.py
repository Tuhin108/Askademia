import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
from datetime import datetime

# Core libraries
import streamlit as st
import pandas as pd
try:
    from pypdf import PdfReader
except ImportError:
    # Fallback or handle missing pypdf gracefully
    PdfReader = None
import fitz  # PyMuPDF for better PDF handling
import pdfplumber
import requests
import arxiv
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    GENAI_AVAILABLE = False

# Security and validation
import hashlib
import re
from functools import wraps
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SecurityManager:
    """Handles security and validation for the AI agent"""
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitize user input to prevent injection attacks"""
        if not isinstance(text, str):
            return ""
        
        # Remove potential harmful patterns
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'on\w+\s*=', '', text, flags=re.IGNORECASE)
        
        # Limit length to prevent DoS
        return text[:10000]
    
    @staticmethod
    def validate_file_type(file_path: str) -> bool:
        """Validate that the file is a PDF"""
        return file_path.lower().endswith('.pdf')
    
    @staticmethod
    def rate_limit(max_calls: int = 60, period: int = 60):
        """Rate limiting decorator"""
        calls = []
        
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                now = time.time()
                calls[:] = [call for call in calls if call > now - period]
                
                if len(calls) >= max_calls:
                    raise Exception(f"Rate limit exceeded: {max_calls} calls per {period} seconds")
                
                calls.append(now)
                return func(*args, **kwargs)
            return wrapper
        return decorator

class DocumentProcessor:
    """Handles document ingestion and processing with multi-modal capabilities"""
    
    def __init__(self):
        self.supported_formats = ['.pdf']
        self.security_manager = SecurityManager()
    
    def extract_text_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text using PyMuPDF with structure preservation"""
        try:
            doc = fitz.open(pdf_path)
            extracted_data = {
                'title': '',
                'abstract': '',
                'sections': [],
                'tables': [],
                'figures': [],
                'references': [],
                'full_text': '',
                'metadata': {}
            }
            
            full_text_parts = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text with formatting
                text = page.get_text()  # type: ignore
                full_text_parts.append(text)

                # Extract images and figures
                image_list = page.get_images()  # type: ignore
                for img_index, img in enumerate(image_list):
                    extracted_data['figures'].append({
                        'page': page_num + 1,
                        'index': img_index,
                        'description': f"Figure on page {page_num + 1}"
                    })

                # Try to extract tables (basic implementation)
                tables = page.find_tables()  # type: ignore
                for table in tables:
                    try:
                        table_data = table.extract()
                        extracted_data['tables'].append({
                            'page': page_num + 1,
                            'data': table_data,
                            'description': f"Table on page {page_num + 1}"
                        })
                    except:
                        continue
            
            extracted_data['full_text'] = '\n'.join(full_text_parts)
            
            # Extract structured information
            self._extract_document_structure(extracted_data)
            
            doc.close()
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error extracting with PyMuPDF: {e}")
            return self._fallback_extraction(pdf_path)
    
    def extract_text_pdfplumber(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text using pdfplumber for better table extraction"""
        try:
            extracted_data = {
                'title': '',
                'abstract': '',
                'sections': [],
                'tables': [],
                'figures': [],
                'references': [],
                'full_text': '',
                'metadata': {}
            }
            
            with pdfplumber.open(pdf_path) as pdf:
                full_text_parts = []
                
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        full_text_parts.append(text)
                    
                    # Extract tables
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if table:
                            extracted_data['tables'].append({
                                'page': page_num + 1,
                                'data': table,
                                'description': f"Table {table_idx + 1} on page {page_num + 1}"
                            })
                
                extracted_data['full_text'] = '\n'.join(full_text_parts)
                self._extract_document_structure(extracted_data)
                
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error extracting with pdfplumber: {e}")
            return self._fallback_extraction(pdf_path)
    
    def _fallback_extraction(self, pdf_path: str) -> Dict[str, Any]:
        """Fallback extraction using PyPDF2"""
        try:
            if PdfReader is None:
                raise ImportError("PyPDF2 not available")
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                
                text_parts = []
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text is not None:
                        text_parts.append(page_text)
                
                full_text = '\n'.join(text_parts)
                
                extracted_data = {
                    'title': '',
                    'abstract': '',
                    'sections': [],
                    'tables': [],
                    'figures': [],
                    'references': [],
                    'full_text': full_text,
                    'metadata': pdf_reader.metadata or {}
                }
                
                self._extract_document_structure(extracted_data)
                return extracted_data
                
        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}")
            return {
                'title': 'Extraction Failed',
                'abstract': '',
                'sections': [],
                'tables': [],
                'figures': [],
                'references': [],
                'full_text': 'Failed to extract text from PDF',
                'metadata': {}
            }
    
    def _extract_document_structure(self, extracted_data: Dict[str, Any]):
        """Extract document structure from full text"""
        text = extracted_data['full_text']
        lines = text.split('\n')
        
        # Extract title (usually first non-empty line or largest text)
        for line in lines[:10]:
            if line.strip() and len(line.strip()) > 10:
                extracted_data['title'] = line.strip()
                break
        
        # Extract abstract
        abstract_start = -1
        abstract_end = -1
        
        for i, line in enumerate(lines):
            if re.search(r'\babstract\b', line.lower()):
                abstract_start = i
            elif abstract_start != -1 and re.search(r'\b(introduction|1\.)', line.lower()):
                abstract_end = i
                break
        
        if abstract_start != -1:
            abstract_lines = lines[abstract_start:abstract_end if abstract_end != -1 else abstract_start + 10]
            extracted_data['abstract'] = ' '.join(abstract_lines).strip()
        
        # Extract sections (basic implementation)
        current_section = ""
        section_content = []
        
        for line in lines:
            # Check if line is a section header
            if re.match(r'^\d+\.?\s+[A-Z]', line.strip()) or re.match(r'^[A-Z\s]+$', line.strip()):
                if current_section and section_content:
                    extracted_data['sections'].append({
                        'title': current_section,
                        'content': ' '.join(section_content)
                    })
                current_section = line.strip()
                section_content = []
            else:
                section_content.append(line.strip())
        
        # Add last section
        if current_section and section_content:
            extracted_data['sections'].append({
                'title': current_section,
                'content': ' '.join(section_content)
            })
        
        # Extract references (basic implementation)
        ref_start = -1
        for i, line in enumerate(lines):
            if re.search(r'\breferences\b|\bbibliography\b', line.lower()):
                ref_start = i
                break
        
        if ref_start != -1:
            ref_lines = lines[ref_start:]
            extracted_data['references'] = [line.strip() for line in ref_lines if line.strip()]
    
    def process_document(self, pdf_path: str) -> Dict[str, Any]:
        """Main document processing pipeline"""
        if not self.security_manager.validate_file_type(pdf_path):
            raise ValueError("Invalid file type. Only PDF files are supported.")
        
        logger.info(f"Processing document: {pdf_path}")
        
        # Try PyMuPDF first (best for structure), then pdfplumber (good for tables)
        try:
            result = self.extract_text_pymupdf(pdf_path)
            if result['full_text'].strip():
                return result
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}")
        
        try:
            result = self.extract_text_pdfplumber(pdf_path)
            if result['full_text'].strip():
                return result
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        
        # Final fallback
        return self._fallback_extraction(pdf_path)

class ArxivIntegration:
    """Handles ArXiv API integration for paper lookup"""
    
    def __init__(self):
        self.client = arxiv.Client()
        self.security_manager = SecurityManager()
    
    @SecurityManager.rate_limit(max_calls=10, period=60)
    def search_papers(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search ArXiv papers based on query"""
        try:
            query = self.security_manager.sanitize_input(query)
            
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = []
            for result in self.client.results(search):
                pdf_url = getattr(result, 'pdf_url', None)
                if not pdf_url:
                    logger.warning(f"No PDF URL found for paper: {result.title}")
                    continue

                papers.append({
                    'title': result.title,
                    'authors': [author.name for author in result.authors],
                    'summary': result.summary,
                    'published': result.published.strftime('%Y-%m-%d'),
                    'pdf_url': pdf_url,
                    'entry_id': result.entry_id,
                    'categories': result.categories
                })
            
            return papers
            
        except Exception as e:
            logger.error(f"ArXiv search error: {e}")
            return []
    
    def download_paper(self, paper_url: str, download_dir: str = "downloads") -> Optional[str]:
        """Download paper from ArXiv"""
        try:
            logger.info(f"Starting download process for URL: {paper_url}")

            # Validate URL
            if not paper_url or not isinstance(paper_url, str):
                logger.error(f"Invalid URL provided: {paper_url} (type: {type(paper_url)})")
                return None

            if not paper_url.startswith('http'):
                logger.error(f"URL does not start with http: {paper_url}")
                return None

            # Check if it's an ArXiv PDF URL
            if 'arxiv.org' not in paper_url.lower() or not paper_url.lower().endswith('.pdf'):
                logger.warning(f"URL may not be a valid ArXiv PDF: {paper_url}")

            os.makedirs(download_dir, exist_ok=True)
            logger.info(f"Download directory ensured: {download_dir}")

            # Add headers to avoid being blocked as a bot
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/pdf,*/*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://arxiv.org/',
                'Connection': 'keep-alive'
            }

            logger.info(f"Sending HTTP request to: {paper_url}")
            response = requests.get(paper_url, headers=headers, timeout=60, stream=True, allow_redirects=True)
            logger.info(f"Response status code: {response.status_code}")

            response.raise_for_status()

            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            logger.info(f"Content-Type header: {content_type}")

            if 'pdf' not in content_type and 'application/octet-stream' not in content_type and 'application/pdf' not in content_type:
                logger.warning(f"Unexpected content type: {content_type} for URL: {paper_url}")
                # Still try to download as some servers don't set content-type properly

            # Check content length
            content_length = response.headers.get('content-length')
            if content_length:
                content_length = int(content_length)
                logger.info(f"Expected content length: {content_length} bytes")
                if content_length == 0:
                    logger.error("Content length is 0, aborting download")
                    return None

            # Generate filename from URL or timestamp
            if 'arxiv.org/pdf/' in paper_url:
                # Extract arxiv ID from URL
                arxiv_id = paper_url.split('/pdf/')[-1].split('.pdf')[0].replace('/', '_')
                filename = f"arxiv_{arxiv_id}.pdf"
            else:
                filename = f"arxiv_paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

            file_path = os.path.join(download_dir, filename)
            logger.info(f"Will save to: {file_path}")

            downloaded_size = 0
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)

            logger.info(f"Downloaded {downloaded_size} bytes")

            # Verify file was downloaded and has content
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                actual_size = os.path.getsize(file_path)
                logger.info(f"Successfully downloaded paper to: {file_path} (size: {actual_size} bytes)")

                # Additional validation: check if file starts with PDF header
                with open(file_path, 'rb') as f:
                    header = f.read(4)
                    if header != b'%PDF':
                        logger.warning(f"Downloaded file does not have PDF header: {header}")
                        # Still return it, as some PDFs might have different headers

                return file_path
            else:
                logger.error(f"Download failed - file is empty or missing: {file_path}")
                if os.path.exists(file_path):
                    os.remove(file_path)
                return None

        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout error downloading paper: {e}")
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error downloading paper: {e}")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error downloading paper: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error downloading paper: {e}")
            return None
        except OSError as e:
            logger.error(f"File system error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading paper: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

class LLMInterface:
    """Handles LLM API integration with context management and optimization"""
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        if not GENAI_AVAILABLE:
            raise ImportError("Google Generative AI library is not available. Please install it with: pip install google-generativeai")

        # Set the API key for the session
        os.environ["GOOGLE_API_KEY"] = api_key
        self.model = model
        self.max_tokens = 4000  # Conservative limit for context
        self.security_manager = SecurityManager()
        
        # We'll initialize the client lazily when needed
        self.client = None
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text (approximate for Gemini)"""
        # Rough approximation: 1 token ≈ 4 characters for Gemini
        return len(text) // 4

    def truncate_context(self, context: str, max_chars: Optional[int] = None) -> str:
        """Truncate context to fit character limit"""
        if max_chars is None:
            max_chars = self.max_tokens * 4  # Convert token limit to character limit

        if len(context) <= max_chars:
            return context

        return context[:max_chars]
    
    @SecurityManager.rate_limit(max_calls=50, period=60)
    def query_document(self, query: str, document_data: Dict[str, Any],
                      query_type: str = "general") -> str:
        """Query document with optimized context handling"""
        try:
            query = self.security_manager.sanitize_input(query)

            # Build context based on query type
            context = self._build_context(document_data, query_type, query)

            # Optimize context length
            context = self.truncate_context(context)

            # Create optimized prompt
            system_prompt = self._get_system_prompt(query_type)
            full_prompt = f"{system_prompt}\n\nContext: {context}\n\nQuery: {query}"

            # Initialize the client if needed
            if self.client is None:
                try:
                    if genai is None:
                        raise ImportError("Google Generative AI library not available")

                    # Configure the API key
                    if hasattr(genai, 'configure'):
                        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))  # type: ignore
                    else:
                        logger.warning("genai.configure not available, trying alternative initialization")

                    # Initialize the model
                    if hasattr(genai, 'GenerativeModel'):
                        self.client = genai.GenerativeModel(model_name=self.model)  # type: ignore
                    elif hasattr(genai, 'get_model'):
                        self.client = genai.get_model(model_name=self.model)  # type: ignore
                    else:
                        # Try alternative initialization methods
                        try:
                            self.client = genai.Model(model_name=self.model)  # type: ignore
                        except AttributeError:
                            # Last resort - try to access the model directly
                            if hasattr(genai, self.model.replace('-', '_').replace('.', '_')):
                                self.client = getattr(genai, self.model.replace('-', '_').replace('.', '_'))
                            else:
                                raise AttributeError("Could not find a way to initialize the model")

                except ImportError as e:
                    logger.error(f"Failed to import Google Generative AI: {e}")
                    return "Google Generative AI library is not available. Please install it with: pip install google-generativeai"
                except Exception as e:
                    logger.error(f"Failed to initialize Gemini client: {e}")
                    return f"Failed to initialize AI model: {str(e)}"

            # Generate response
            try:
                response = self.client.generate_content(full_prompt)

                if hasattr(response, 'text') and response.text:
                    return response.text.strip()
                elif isinstance(response, str):
                    return response.strip()
                else:
                    logger.warning(f"Unexpected response format: {type(response)}")
                    return "No response generated from AI model"

            except Exception as e:
                logger.error(f"Error generating content with Gemini: {e}")
                return f"AI model error: {str(e)}"

        except Exception as e:
            logger.error(f"LLM query error: {e}")
            return f"I apologize, but I encountered an error processing your query: {str(e)}"
    
    def _build_context(self, document_data: Dict[str, Any], 
                      query_type: str, query: str) -> str:
        """Build optimized context based on query type"""
        context_parts = []
        
        # Always include title and abstract
        if document_data.get('title'):
            context_parts.append(f"Title: {document_data['title']}")
        
        if document_data.get('abstract'):
            context_parts.append(f"Abstract: {document_data['abstract']}")
        
        # Add relevant sections based on query type
        if query_type == "methodology" and document_data.get('sections'):
            method_sections = [s for s in document_data['sections'] 
                             if any(term in s['title'].lower() 
                                   for term in ['method', 'approach', 'algorithm', 'procedure'])]
            for section in method_sections[:2]:  # Limit to 2 sections
                context_parts.append(f"Section - {section['title']}: {section['content']}")
        
        elif query_type == "results" and document_data.get('sections'):
            result_sections = [s for s in document_data['sections'] 
                             if any(term in s['title'].lower() 
                                   for term in ['result', 'evaluation', 'experiment', 'performance'])]
            for section in result_sections[:2]:
                context_parts.append(f"Section - {section['title']}: {section['content']}")
            
            # Include tables for numerical results
            if document_data.get('tables'):
                for table in document_data['tables'][:2]:
                    context_parts.append(f"Table from page {table['page']}: {table['description']}")
        
        elif query_type == "conclusion" and document_data.get('sections'):
            conclusion_sections = [s for s in document_data['sections'] 
                                 if any(term in s['title'].lower() 
                                       for term in ['conclusion', 'summary', 'discussion'])]
            for section in conclusion_sections:
                context_parts.append(f"Section - {section['title']}: {section['content']}")
        
        else:  # General query - include most relevant content
            # Use keyword matching to find relevant sections
            query_lower = query.lower()
            relevant_sections = []
            
            for section in document_data.get('sections', []):
                section_text = f"{section['title']} {section['content']}".lower()
                relevance_score = sum(1 for word in query_lower.split() 
                                    if word in section_text and len(word) > 3)
                
                if relevance_score > 0:
                    relevant_sections.append((section, relevance_score))
            
            # Sort by relevance and take top sections
            relevant_sections.sort(key=lambda x: x[1], reverse=True)
            for section, _ in relevant_sections[:3]:
                context_parts.append(f"Section - {section['title']}: {section['content']}")
        
        # Include full text as fallback (truncated)
        if not any('Section' in part for part in context_parts):
            context_parts.append(f"Document content: {document_data.get('full_text', '')}")
        
        return '\n\n'.join(context_parts)
    
    def _get_system_prompt(self, query_type: str) -> str:
        """Get system prompt based on query type"""
        base_prompt = """You are an expert AI assistant specialized in analyzing academic documents. 
        Provide accurate, detailed, and well-structured answers based on the provided document context."""
        
        if query_type == "methodology":
            return base_prompt + " Focus on explaining the methods, approaches, and procedures described in the document."
        elif query_type == "results":
            return base_prompt + " Focus on presenting the results, evaluations, and performance metrics from the document."
        elif query_type == "conclusion":
            return base_prompt + " Focus on summarizing the conclusions, key findings, and implications from the document."
        else:
            return base_prompt + " Answer the query comprehensively using the provided document context."

    def function_call_arxiv_lookup(self, query: str, arxiv_integration: ArxivIntegration) -> str:
        """Function calling capability for ArXiv paper lookup"""
        try:
            papers = arxiv_integration.search_papers(query)
            
            if not papers:
                return f"No papers found for query: {query}"
            
            response_parts = [f"Found {len(papers)} relevant papers for '{query}':\n"]
            
            for i, paper in enumerate(papers, 1):
                summary = paper.get('summary', 'No summary available')
                summary_text = summary[:200] + '...' if summary and len(summary) > 200 else summary or 'No summary available'
                response_parts.append(
                    f"{i}. **{paper['title']}**\n"
                    f"   Authors: {', '.join(paper['authors'])}\n"
                    f"   Published: {paper['published']}\n"
                    f"   Categories: {', '.join(paper['categories'])}\n"
                    f"   Summary: {summary_text}\n"
                    f"   PDF: {paper.get('pdf_url', 'N/A')}\n"
                )
            
            return '\n'.join(response_parts)
            
        except Exception as e:
            return f"Error searching ArXiv: {str(e)}"

class DocumentQAAgent:
    """Main AI Agent class that orchestrates all components"""
    
    def __init__(self, gemini_api_key: str):
        self.document_processor = DocumentProcessor()
        self.arxiv_integration = ArxivIntegration()
        self.llm_interface = LLMInterface(gemini_api_key)
        self.processed_documents = {}
        self.security_manager = SecurityManager()
        
        logger.info("Document Q&A Agent initialized successfully with Gemini API")
    
    def ingest_document(self, pdf_path: str) -> str:
        """Ingest a PDF document into the system"""
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"File not found: {pdf_path}")
            
            # Generate document ID
            doc_id = hashlib.md5(pdf_path.encode()).hexdigest()[:8]
            
            # Process document
            document_data = self.document_processor.process_document(pdf_path)
            document_data['file_path'] = pdf_path
            document_data['doc_id'] = doc_id
            document_data['processed_at'] = datetime.now().isoformat()
            
            # Store processed document
            self.processed_documents[doc_id] = document_data
            
            logger.info(f"Document ingested successfully: {pdf_path} (ID: {doc_id})")
            return doc_id
            
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            raise
    
    def query_document(self, query: str, doc_id: Optional[str] = None) -> str:
        """Query a specific document or all documents"""
        try:
            query = self.security_manager.sanitize_input(query)
            
            if not self.processed_documents:
                return "No documents have been ingested yet. Please upload a PDF document first."
            
            # Determine query type
            query_type = self._classify_query(query)
            
            if doc_id:
                if doc_id not in self.processed_documents:
                    return f"Document with ID {doc_id} not found."
                
                document_data = self.processed_documents[doc_id]
                return self.llm_interface.query_document(query, document_data, query_type)
            
            else:
                # Query all documents
                responses = []
                for doc_id, document_data in self.processed_documents.items():
                    response = self.llm_interface.query_document(query, document_data, query_type)
                    responses.append(f"**Document: {document_data.get('title', doc_id)}**\n{response}")
                
                return '\n\n---\n\n'.join(responses)
                
        except Exception as e:
            logger.error(f"Document query failed: {e}")
            return f"I apologize, but I encountered an error processing your query: {str(e)}"
    
    def _classify_query(self, query: str) -> str:
        """Classify query type for optimized context selection"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['method', 'approach', 'algorithm', 'procedure', 'how']):
            return "methodology"
        elif any(term in query_lower for term in ['result', 'performance', 'accuracy', 'f1', 'score', 'evaluation']):
            return "results"
        elif any(term in query_lower for term in ['conclusion', 'summary', 'findings', 'implications']):
            return "conclusion"
        else:
            return "general"
    
    def search_arxiv_papers(self, query: str) -> str:
        """Search and provide information about ArXiv papers"""
        try:
            return self.llm_interface.function_call_arxiv_lookup(query, self.arxiv_integration)
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            return f"Error searching ArXiv papers: {str(e)}"
    
    def download_arxiv_paper(self, paper_url: str) -> Optional[str]:
        """Download a paper from ArXiv and ingest it"""
        try:
            logger.info(f"Starting download of ArXiv paper from: {paper_url}")
            file_path = self.arxiv_integration.download_paper(paper_url)
            if file_path:
                logger.info(f"Download successful, file saved to: {file_path}")
                doc_id = self.ingest_document(file_path)
                logger.info(f"Document ingested successfully with ID: {doc_id}")
                return doc_id
            else:
                logger.error("Download failed - no file path returned")
                return None
        except Exception as e:
            logger.error(f"ArXiv paper download failed: {e}")
            return None
    
    def get_document_summary(self, doc_id: str) -> str:
        """Get a summary of a processed document"""
        if doc_id not in self.processed_documents:
            return f"Document with ID {doc_id} not found."
        
        doc_data = self.processed_documents[doc_id]
        
        summary_parts = [
            f"**Title:** {doc_data.get('title', 'N/A')}",
            f"**Document ID:** {doc_id}",
            f"**Processed:** {doc_data.get('processed_at', 'N/A')}",
            f"**Sections:** {len(doc_data.get('sections', []))}",
            f"**Tables:** {len(doc_data.get('tables', []))}",
            f"**Figures:** {len(doc_data.get('figures', []))}",
        ]
        
        if doc_data.get('abstract'):
            summary_parts.append(f"**Abstract:** {doc_data['abstract'][:300]}...")
        
        return '\n'.join(summary_parts)
    
    def list_documents(self) -> str:
        """List all processed documents"""
        if not self.processed_documents:
            return "No documents have been processed yet."
        
        doc_list = []
        for doc_id, doc_data in self.processed_documents.items():
            doc_list.append(f"- **{doc_id}**: {doc_data.get('title', 'Untitled Document')}")
        
        return "**Processed Documents:**\n" + '\n'.join(doc_list)

if __name__ == "__main__":
    # This would be used for testing

    pass
