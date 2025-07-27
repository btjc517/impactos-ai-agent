"""
Query and Q&A system for ImpactOS AI Layer MVP Phase One.

This module handles natural language queries, performs hybrid search using
FAISS vectors + SQL queries, and generates cited answers using GPT-4.
"""

import sqlite3
import os
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

# AI and ML imports
from sentence_transformers import SentenceTransformer
import openai
from openai import OpenAI
import numpy as np

# Local imports
from schema import DatabaseSchema

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuerySystem:
    """Handles natural language queries and generates cited answers."""
    
    def __init__(self, db_path: str = "db/impactos.db"):
        """
        Initialize query system.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.db_schema = DatabaseSchema(db_path)
        
        # Initialize AI components
        self.openai_client = self._initialize_openai()
        self.embedding_model = self._initialize_embedding_model()
        
        # Query processing configuration
        self.similarity_threshold = 0.7
        self.max_results = 10
    
    def _initialize_openai(self) -> Optional[OpenAI]:
        """Initialize OpenAI client with API key."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.warning("OPENAI_API_KEY not found. GPT-4 orchestration disabled.")
            return None
        
        try:
            client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            return None
    
    def _initialize_embedding_model(self) -> Optional[SentenceTransformer]:
        """Initialize sentence transformer for query embeddings."""
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model initialized successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            return None
    
    def query(self, question: str) -> str:
        """
        Process natural language query and return cited answer.
        
        Args:
            question: Natural language question
            
        Returns:
            Answer with citations
        """
        try:
            logger.info(f"Processing query: {question}")
            
            # 1. Analyze query intent and extract key terms
            query_analysis = self._analyze_query(question)
            
            # 2. Perform hybrid search (SQL + vector similarity)
            results = self._hybrid_search(question, query_analysis)
            
            # 3. Generate cited answer using GPT-4 (if available)
            if self.openai_client and results:
                answer = self._generate_gpt_answer(question, results)
            else:
                answer = self._generate_fallback_answer(question, results)
            
            logger.info("Query processing completed")
            return answer
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Error processing query: {e}"
    
    def _analyze_query(self, question: str) -> Dict[str, Any]:
        """Analyze query to extract intent and key terms."""
        question_lower = question.lower()
        
        # Simple keyword-based analysis
        analysis = {
            'intent': 'general',
            'categories': [],
            'metrics': [],
            'time_references': [],
            'aggregations': []
        }
        
        # Detect categories
        category_keywords = {
            'volunteering': ['volunteer', 'volunteering', 'community service'],
            'donations': ['donation', 'charity', 'giving', 'contributed'],
            'carbon': ['carbon', 'co2', 'emission', 'environmental'],
            'procurement': ['procurement', 'supplier', 'local spend']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                analysis['categories'].append(category)
        
        # Detect aggregations
        if any(word in question_lower for word in ['total', 'sum', 'amount', 'how much']):
            analysis['aggregations'].append('sum')
        if any(word in question_lower for word in ['average', 'mean']):
            analysis['aggregations'].append('average')
        if any(word in question_lower for word in ['count', 'how many', 'number']):
            analysis['aggregations'].append('count')
        
        # Detect time references
        if any(word in question_lower for word in ['last year', 'this year', '2024', '2023']):
            analysis['time_references'].append('yearly')
        
        return analysis
    
    def _hybrid_search(self, question: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform hybrid search using SQL queries and vector similarity."""
        try:
            results = []
            
            # 1. SQL-based search for metrics
            sql_results = self._sql_search(analysis)
            results.extend(sql_results)
            
            # 2. Vector similarity search (if embedding model available)
            if self.embedding_model:
                vector_results = self._vector_search(question)
                results.extend(vector_results)
            
            # 3. Deduplicate and rank results
            results = self._deduplicate_results(results)
            
            logger.info(f"Hybrid search found {len(results)} relevant results")
            return results[:self.max_results]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    def _sql_search(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search database using SQL queries based on query analysis."""
        try:
            results = []
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row  # Enable column access by name
                cursor = conn.cursor()
                
                # Build WHERE clause based on categories
                where_conditions = []
                params = []
                
                if analysis['categories']:
                    category_placeholders = ','.join(['?' for _ in analysis['categories']])
                    where_conditions.append(f"im.metric_category IN ({category_placeholders})")
                    params.extend(analysis['categories'])
                
                where_clause = ' AND '.join(where_conditions) if where_conditions else '1=1'
                
                # Query for metrics with aggregation if requested
                if 'sum' in analysis['aggregations']:
                    query = f"""
                        SELECT 
                            im.metric_category,
                            im.metric_name,
                            SUM(im.metric_value) as total_value,
                            im.metric_unit,
                            COUNT(*) as count,
                            s.filename
                        FROM impact_metrics im
                        JOIN sources s ON im.source_id = s.id
                        WHERE {where_clause}
                        GROUP BY im.metric_category, im.metric_name, im.metric_unit
                        ORDER BY total_value DESC
                    """
                else:
                    query = f"""
                        SELECT 
                            im.metric_name,
                            im.metric_value,
                            im.metric_unit,
                            im.metric_category,
                            im.context_description,
                            im.extraction_confidence,
                            s.filename,
                            s.processed_timestamp
                        FROM impact_metrics im
                        JOIN sources s ON im.source_id = s.id
                        WHERE {where_clause}
                        ORDER BY im.extraction_confidence DESC, im.metric_value DESC
                    """
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                for row in rows:
                    result = {
                        'type': 'sql_metric',
                        'data': dict(row),
                        'relevance_score': 0.9  # High relevance for exact category matches
                    }
                    results.append(result)
                
                logger.info(f"SQL search found {len(results)} results")
                return results
                
        except Exception as e:
            logger.error(f"Error in SQL search: {e}")
            return []
    
    def _vector_search(self, question: str) -> List[Dict[str, Any]]:
        """Search using vector similarity on embeddings."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([question])
            
            # Get all stored embeddings
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        e.text_chunk,
                        e.chunk_type,
                        e.metric_id,
                        im.metric_name,
                        im.metric_value,
                        im.metric_unit,
                        im.metric_category,
                        s.filename
                    FROM embeddings e
                    LEFT JOIN impact_metrics im ON e.metric_id = im.id
                    LEFT JOIN sources s ON im.source_id = s.id
                    WHERE e.metric_id IS NOT NULL
                """)
                
                rows = cursor.fetchall()
                
                results = []
                for row in rows:
                    # For now, use simple text matching as similarity
                    # In a full implementation, we'd store and compare actual vectors
                    text_chunk = row['text_chunk'].lower()
                    question_lower = question.lower()
                    
                    # Simple word overlap similarity
                    question_words = set(question_lower.split())
                    chunk_words = set(text_chunk.split())
                    overlap = len(question_words.intersection(chunk_words))
                    similarity = overlap / len(question_words) if question_words else 0
                    
                    if similarity > 0.1:  # Basic threshold
                        result = {
                            'type': 'vector_match',
                            'data': dict(row),
                            'relevance_score': similarity
                        }
                        results.append(result)
                
                # Sort by relevance
                results.sort(key=lambda x: x['relevance_score'], reverse=True)
                
                logger.info(f"Vector search found {len(results)} results")
                return results
                
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results and merge information."""
        seen_metrics = set()
        deduplicated = []
        
        for result in results:
            data = result['data']
            # Create a unique key for the metric
            key = (
                data.get('metric_name', ''),
                data.get('metric_category', ''),
                data.get('filename', '')
            )
            
            if key not in seen_metrics:
                seen_metrics.add(key)
                deduplicated.append(result)
        
        return deduplicated
    
    def _generate_gpt_answer(self, question: str, results: List[Dict[str, Any]]) -> str:
        """Generate answer using GPT-4 with citations."""
        try:
            # Prepare context from search results
            context_parts = []
            
            for i, result in enumerate(results[:5]):  # Use top 5 results
                data = result['data']
                
                if result['type'] == 'sql_metric':
                    if 'total_value' in data:
                        context_parts.append(
                            f"[{i+1}] {data['metric_category'].title()}: Total {data['metric_name']} = "
                            f"{data['total_value']} {data['metric_unit']} from {data['count']} records "
                            f"(Source: {data['filename']})"
                        )
                    else:
                        context_parts.append(
                            f"[{i+1}] {data['metric_category'].title()}: {data['metric_name']} = "
                            f"{data['metric_value']} {data['metric_unit']} "
                            f"(Source: {data['filename']}, Confidence: {data['extraction_confidence']:.2f})"
                        )
                else:
                    context_parts.append(
                        f"[{i+1}] {data['text_chunk']} (Source: {data['filename']})"
                    )
            
            context = '\n'.join(context_parts)
            
            prompt = f"""
            You are analyzing social value data to answer questions with accurate citations.
            
            Question: {question}
            
            Available data:
            {context}
            
            Instructions:
            1. Answer the question based ONLY on the provided data
            2. Include specific numbers, values, and units where available
            3. Cite sources using [1], [2], etc. format
            4. If data is insufficient, state that clearly
            5. Be concise but comprehensive
            
            Format your answer as:
            [Answer with specific data and citations]
            
            Sources:
            [List of sources referenced]
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
            
        except Exception as e:
            logger.error(f"Error generating GPT answer: {e}")
            return self._generate_fallback_answer(question, results)
    
    def _generate_fallback_answer(self, question: str, results: List[Dict[str, Any]]) -> str:
        """Generate basic answer when GPT-4 is not available."""
        if not results:
            return (
                "I couldn't find specific data to answer your question. "
                "This might be because:\n"
                "1. No relevant data has been ingested yet\n"
                "2. The question doesn't match available metrics\n"
                "3. Try rephrasing your question or ingest more data files"
            )
        
        answer_parts = [f"Based on the available data, here's what I found:\n"]
        
        for i, result in enumerate(results[:5]):
            data = result['data']
            
            if result['type'] == 'sql_metric':
                if 'total_value' in data:
                    answer_parts.append(
                        f"{i+1}. {data['metric_category'].title()}: "
                        f"Total {data['metric_name']} = {data['total_value']} {data['metric_unit']} "
                        f"from {data['count']} records"
                    )
                else:
                    answer_parts.append(
                        f"{i+1}. {data['metric_category'].title()}: "
                        f"{data['metric_name']} = {data['metric_value']} {data['metric_unit']}"
                    )
            else:
                answer_parts.append(f"{i+1}. {data['text_chunk']}")
        
        # Add sources
        sources = set()
        for result in results:
            filename = result['data'].get('filename')
            if filename:
                sources.add(filename)
        
        if sources:
            answer_parts.append(f"\nSources: {', '.join(sources)}")
        
        return '\n'.join(answer_parts)


def query_data(question: str, db_path: str = "db/impactos.db") -> str:
    """Convenience function for querying data."""
    query_system = QuerySystem(db_path)
    return query_system.query(question)


if __name__ == "__main__":
    # Test query system
    import sys
    
    if len(sys.argv) > 1:
        question = ' '.join(sys.argv[1:])
        answer = query_data(question)
        print(f"\nQuestion: {question}")
        print(f"Answer: {answer}")
    else:
        print("Usage: python query.py <question>")
        print("Example: python query.py 'How much was donated to charity?'") 