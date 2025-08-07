"""
Query and Q&A system for ImpactOS AI Layer MVP Phase One.

This module handles natural language queries with enhanced functionality:
1. Replacing hard top-10 limit with relevance threshold filtering
2. Query-type specific result limits (higher for aggregations)
3. Two-stage processing: retrieve all relevant, then filter for GPT-4
4. Progressive summarization for large result sets

Features focus on accuracy and completeness over artificial speed limits.
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
from frameworks import FrameworkMapper
from vector_search import FAISSVectorSearch
from config import get_config, get_config_manager

# Setup logging is configured at the entrypoint; avoid per-module basicConfig
logger = logging.getLogger(__name__)


class QuerySystem:
    """Enhanced query system with intelligent result filtering."""
    
    def __init__(self, db_path: str = "db/impactos.db"):
        """
        Initialize query system.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.db_schema = DatabaseSchema(db_path)
        self.framework_mapper = FrameworkMapper(db_path)
        
        # Initialize AI components
        self.openai_client = self._initialize_openai()
        self.vector_search = FAISSVectorSearch(db_path)  # Proper FAISS vector search
        
        # Load dynamic configuration (replaces hardcoded values)
        self.config = get_config()
        self.config_manager = get_config_manager()
    
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
            
            # 2. Perform enhanced hybrid search with intelligent limits
            results = self._enhanced_hybrid_search(question, query_analysis)
            
            logger.info(f"Retrieved {len(results)} total relevant results")
            
            # 3. Intelligent filtering and summarization for GPT-4
            filtered_results = self._intelligent_filter_for_gpt(results, query_analysis)
            
            logger.info(f"Filtered to {len(filtered_results)} results for GPT-4")
            
            # 4. Generate cited answer using GPT-4 (if available)
            if self.openai_client and filtered_results:
                answer = self._generate_gpt_answer(question, filtered_results)
            else:
                answer = self._generate_fallback_answer(question, results)
            
            logger.info("Query processing completed")
            return answer
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Error processing query: {e}"
    
    def _analyze_query(self, question: str) -> Dict[str, Any]:
        """Enhanced query analysis to extract intent and key terms."""
        question_lower = question.lower()
        
        analysis = {
            'intent': 'general',
            'categories': [],
            'metrics': [],
            'time_references': [],
            'aggregations': [],
            'query_type': 'descriptive'  # New field for query type classification
        }
        
        # Prefer embedding-based intent detection if available
        cfg = getattr(self.config, 'analysis', None)
        used_embedding_intent = False
        if cfg and getattr(cfg, 'use_embedding_for_intent', False) and self.vector_search and self.vector_search.embedding_model:
            try:
                query_vec = self.vector_search.embedding_model.encode([question])[0]
                # Prepare prototypes
                category_texts = list(cfg.category_descriptions.values()) if getattr(cfg, 'category_descriptions', None) else []
                category_labels = list(cfg.category_descriptions.keys()) if getattr(cfg, 'category_descriptions', None) else []
                agg_texts = list(cfg.aggregation_descriptions.values()) if getattr(cfg, 'aggregation_descriptions', None) else []
                agg_labels = list(cfg.aggregation_descriptions.keys()) if getattr(cfg, 'aggregation_descriptions', None) else []
                
                # Encode prototypes
                cat_vecs = self.vector_search.embedding_model.encode(category_texts) if category_texts else []
                agg_vecs = self.vector_search.embedding_model.encode(agg_texts) if agg_texts else []
                
                # Cosine similarity
                def cos_sim(a, b):
                    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                
                # Categories
                for label, proto_vec in zip(category_labels, cat_vecs):
                    sim = cos_sim(query_vec, proto_vec)
                    if sim >= cfg.category_min_similarity:
                        analysis['categories'].append(label)
                
                # Aggregations
                for label, proto_vec in zip(agg_labels, agg_vecs):
                    sim = cos_sim(query_vec, proto_vec)
                    if sim >= cfg.aggregation_min_similarity:
                        analysis['aggregations'].append(label)
                used_embedding_intent = True
            except Exception as e:
                logger.warning(f"Embedding-based intent detection failed, falling back to keywords: {e}")
                used_embedding_intent = False
        
        # Fallback or complement with keyword matching
        if not used_embedding_intent:
            # Detect categories with expanded keywords (from config)
            category_keywords = getattr(self.config.analysis, 'category_keywords', {})
            for category, keywords in category_keywords.items():
                if any(keyword in question_lower for keyword in keywords):
                    analysis['categories'].append(category)
            
            # Detect aggregations (from config)
            aggregation_patterns = getattr(self.config.analysis, 'aggregation_patterns', {})
            for agg_type, patterns in aggregation_patterns.items():
                if any(pattern in question_lower for pattern in patterns):
                    analysis['aggregations'].append(agg_type)
        
        # Classify query type for intelligent processing
        if analysis['aggregations']:
            analysis['query_type'] = 'aggregation'
        elif any(word in question_lower for word in ['what', 'which', 'list', 'show', 'available']):
            analysis['query_type'] = 'descriptive'
        elif any(word in question_lower for word in ['how', 'why', 'when', 'where']):
            analysis['query_type'] = 'analytical'
        
        # Detect time references (from config)
        time_patterns = getattr(self.config.analysis, 'time_patterns', [])
        for pattern in time_patterns:
            if pattern in question_lower:
                analysis['time_references'].append(pattern)
        
        return analysis
    
    def _enhanced_hybrid_search(self, question: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhanced hybrid search with intelligent result limits."""
        try:
            results = []
            
            # 1. SQL-based search for metrics with enhanced queries
            sql_results = self._enhanced_sql_search(analysis)
            results.extend(sql_results)
            
            # 2. Proper FAISS vector similarity search
            vector_results = self._faiss_vector_search(question, analysis)
            results.extend(vector_results)
            
            # 3. Filter by relevance threshold instead of hard limit
            relevant_results = self._filter_by_relevance(results)
            
            # 4. Apply query-type specific limits
            final_results = self._apply_query_specific_limits(relevant_results, analysis)
            
            logger.info(f"Enhanced hybrid search: {len(results)} initial -> {len(relevant_results)} relevant -> {len(final_results)} final")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in enhanced hybrid search: {e}")
            return []
    
    def _enhanced_sql_search(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhanced SQL search with better aggregation handling."""
        try:
            results = []
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Build WHERE clause based on categories with fallback for broad search
                where_conditions = []
                params = []
                
                if analysis['categories']:
                    category_placeholders = ','.join(['?' for _ in analysis['categories']])
                    where_conditions.append(f"im.metric_category IN ({category_placeholders})")
                    params.extend(analysis['categories'])
                
                # If no specific categories or query type suggests broad search, use looser filtering
                if not where_conditions or analysis['query_type'] in ['descriptive', 'analytical']:
                    # For broad queries, search across all relevant text fields
                    broad_search_terms = []
                    if 'training' in analysis.get('categories', []) or any('train' in word for word in analysis.get('categories', [])):
                        broad_search_terms.extend(['training', 'education', 'development', 'course'])
                    elif 'employee' in analysis.get('categories', []):
                        broad_search_terms.extend(['employee', 'staff', 'personnel', 'assistance'])
                    elif 'diversity' in analysis.get('categories', []):
                        broad_search_terms.extend(['diversity', 'inclusion', 'gender', 'equality'])
                    
                    if broad_search_terms:
                        text_conditions = []
                        for term in broad_search_terms:
                            text_conditions.extend([
                                f"LOWER(im.metric_name) LIKE ?",
                                f"LOWER(im.context_description) LIKE ?",
                                f"LOWER(im.source_column_name) LIKE ?"
                            ])
                            params.extend([f'%{term}%', f'%{term}%', f'%{term}%'])
                        
                        if where_conditions:
                            where_conditions.append(f"({' OR '.join(text_conditions)})")
                        else:
                            where_conditions.append(f"({' OR '.join(text_conditions)})")
                
                where_clause = ' AND '.join(where_conditions) if where_conditions else '1=1'
                
                # Enhanced aggregation queries
                if 'sum' in analysis['aggregations'] or analysis['query_type'] == 'aggregation':
                    # First get individual metrics for detailed analysis
                    individual_query = f"""
                        SELECT 
                            im.metric_name,
                            im.metric_value,
                            im.metric_unit,
                            im.metric_category,
                            im.context_description,
                            im.extraction_confidence,
                            im.source_sheet_name,
                            im.source_column_name,
                            im.source_cell_reference,
                            im.source_formula,
                            im.verification_status,
                            s.filename,
                            s.processed_timestamp
                        FROM impact_metrics im
                        JOIN sources s ON im.source_id = s.id
                        WHERE {where_clause}
                        ORDER BY im.extraction_confidence DESC, im.metric_value DESC
                    """
                    
                    cursor.execute(individual_query, params)
                    individual_rows = cursor.fetchall()
                    
                    for row in individual_rows:
                        result = {
                            'type': 'sql_individual_metric',
                            'data': dict(row),
                            'relevance_score': 0.95  # Very high relevance for individual metrics
                        }
                        results.append(result)
                    
                    # Then get aggregated summaries
                    aggregated_query = f"""
                        SELECT 
                            im.metric_category,
                            im.metric_name,
                            SUM(im.metric_value) as total_value,
                            im.metric_unit,
                            COUNT(*) as count,
                            AVG(im.metric_value) as avg_value,
                            MIN(im.metric_value) as min_value,
                            MAX(im.metric_value) as max_value,
                            GROUP_CONCAT(DISTINCT im.source_cell_reference) as cell_references,
                            GROUP_CONCAT(DISTINCT im.source_column_name) as column_names,
                            GROUP_CONCAT(DISTINCT im.source_sheet_name) as sheet_names,
                            GROUP_CONCAT(DISTINCT s.filename) as filenames
                        FROM impact_metrics im
                        JOIN sources s ON im.source_id = s.id
                        WHERE {where_clause}
                        GROUP BY im.metric_category, im.metric_name, im.metric_unit
                        HAVING COUNT(*) > 1  -- Only include metrics with multiple entries
                        ORDER BY total_value DESC
                    """
                    
                    cursor.execute(aggregated_query, params)
                    agg_rows = cursor.fetchall()
                    
                    for row in agg_rows:
                        result = {
                            'type': 'sql_aggregated_metric',
                            'data': dict(row),
                            'relevance_score': 0.98  # Highest relevance for aggregated results
                        }
                        results.append(result)
                
                else:
                    # Standard detailed query for non-aggregation queries
                    query = f"""
                        SELECT 
                            im.metric_name,
                            im.metric_value,
                            im.metric_unit,
                            im.metric_category,
                            im.context_description,
                            im.extraction_confidence,
                            im.source_sheet_name,
                            im.source_column_name,
                            im.source_cell_reference,
                            im.source_formula,
                            im.verification_status,
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
                
                logger.info(f"Enhanced SQL search found {len(results)} results")
                return results
                
        except Exception as e:
            logger.error(f"Error in enhanced SQL search: {e}")
            return []
    
    def _faiss_vector_search(self, question: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Proper FAISS vector similarity search."""
        try:
            # Use FAISS for proper vector similarity search with dynamic configuration
            query_type = analysis.get('query_type', 'default')
            similarity_threshold = self.config_manager.get_similarity_threshold(query_type)
            
            results = self.vector_search.search(
                query=question,
                k=self.config.query_processing.max_initial_retrieval,
                min_similarity=similarity_threshold
            )
            
            logger.info(f"FAISS vector search found {len(results)} results")
            return results
                
        except Exception as e:
            logger.error(f"Error in FAISS vector search: {e}")
            return []
    
    def _filter_by_relevance(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter results by relevance threshold instead of hard limit."""
        relevant_results = []
        seen_metrics = set()
        
        # Sort all results by relevance score
        sorted_results = sorted(results, key=lambda x: x['relevance_score'], reverse=True)
        
        for result in sorted_results:
            # Apply dynamic relevance threshold
            min_threshold = self.config.vector_search.min_similarity_threshold
            if result['relevance_score'] >= min_threshold:
                data = result['data']
                
                # Create unique key to avoid exact duplicates
                key = (
                    data.get('metric_name', ''),
                    data.get('metric_category', ''),
                    data.get('filename', ''),
                    data.get('metric_value', '')
                )
                
                if key not in seen_metrics:
                    seen_metrics.add(key)
                    relevant_results.append(result)
        
        return relevant_results
    
    def _apply_query_specific_limits(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply dynamic query-type specific result limits."""
        query_type = analysis['query_type']
        limit = self.config_manager.get_result_limit(query_type)
        
        return results[:limit]
    
    def _intelligent_filter_for_gpt(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Intelligently filter results for GPT-4 context window."""
        max_results = self.config.query_processing.max_results_for_gpt
        if len(results) <= max_results:
            return results
        
        # For aggregation queries, prioritize aggregated results and high-value individual metrics
        if analysis['query_type'] == 'aggregation':
            aggregated = [r for r in results if r['type'] == 'sql_aggregated_metric']
            individual = [r for r in results if r['type'] in ['sql_individual_metric', 'sql_metric']]
            faiss_results = [r for r in results if r['type'] == 'faiss_vector_match']
            
            # Take configurable slices of each group
            cfg = self.config.query_processing
            max_aggr = getattr(cfg, 'aggregation_ctx_max_aggregated', 5)
            max_ind = getattr(cfg, 'aggregation_ctx_max_individual', 8)
            max_vec = getattr(cfg, 'aggregation_ctx_max_vector', 2)
            filtered = aggregated[:max_aggr] + individual[:max_ind] + faiss_results[:max_vec]
            filtered = filtered[:max_results]
        else:
            # For other queries, take top results by relevance
            filtered = results[:self.config.query_processing.max_results_for_gpt]
        
        return filtered
    
    def _get_framework_mappings(self, metric_name: str, metric_category: str) -> List[str]:
        """Get framework mappings for a metric."""
        try:
            mappings = self.framework_mapper.map_metric_to_frameworks(metric_name, metric_category)
            framework_list = []
            
            for framework, codes in mappings.items():
                if codes:
                    framework_name = framework.replace('_', ' ').title()
                    framework_list.append(f"{framework_name}: {', '.join(codes)}")
            
            return framework_list
        except Exception as e:
            logger.error(f"Error getting framework mappings: {e}")
            return []
    
    def _format_cell_reference(self, data: Dict[str, Any]) -> str:
        """Format cell reference information for display."""
        parts = []
        
        if data.get('source_sheet_name'):
            parts.append(f"Sheet: {data['source_sheet_name']}")
        
        if data.get('source_cell_reference'):
            parts.append(f"Cell: {data['source_cell_reference']}")
        elif data.get('source_column_name'):
            parts.append(f"Column: {data['source_column_name']}")
        
        if data.get('source_formula'):
            parts.append(f"Formula: {data['source_formula']}")
        
        return "; ".join(parts) if parts else "No specific cell reference"
    
    def _generate_gpt_answer(self, question: str, results: List[Dict[str, Any]]) -> str:
        """Generate answer using GPT-4 with enhanced context."""
        try:
            # Prepare enhanced context from search results
            context_parts = []
            
            # Group results by type for better organization
            aggregated_results = [r for r in results if r['type'] == 'sql_aggregated_metric']
            individual_results = [r for r in results if r['type'] in ['sql_individual_metric', 'sql_metric']]
            vector_results = [r for r in results if r['type'] == 'faiss_vector_match']
            
            # Format aggregated results first (most important for aggregation queries)
            for i, result in enumerate(aggregated_results):
                data = result['data']
                frameworks = self._get_framework_mappings(
                    data.get('metric_name', ''), 
                    data.get('metric_category', '')
                )
                framework_text = f" | Frameworks: {'; '.join(frameworks)}" if frameworks else ""
                
                context_parts.append(
                    f"[{len(context_parts)+1}] AGGREGATED: {data['metric_category'].title()}: "
                    f"Total {data['metric_name']} = {data['total_value']} {data['metric_unit']} "
                    f"(from {data['count']} records, avg: {data.get('avg_value', 'N/A')}) "
                    f"(Sources: {data.get('filenames', 'Unknown')}{framework_text})"
                )
            
            # Then format individual results
            for i, result in enumerate(individual_results[:8]):  # Limit individual results
                data = result['data']
                frameworks = self._get_framework_mappings(
                    data.get('metric_name', ''), 
                    data.get('metric_category', '')
                )
                framework_text = f" | Frameworks: {'; '.join(frameworks)}" if frameworks else ""
                cell_ref = self._format_cell_reference(data)
                
                verification_status = data.get('verification_status', 'pending')
                status_indicator = "✓" if verification_status == 'verified' else "⚠" if verification_status == 'failed' else "?"
                
                context_parts.append(
                    f"[{len(context_parts)+1}] {data['metric_category'].title()}: {data['metric_name']} = "
                    f"{data['metric_value']} {data['metric_unit']} {status_indicator} "
                    f"(Source: {data['filename']} | {cell_ref} | "
                    f"Confidence: {data['extraction_confidence']:.2f}{framework_text})"
                )
            
            # Include vector results to ensure GPT has context even when SQL-based results are absent
            if vector_results:
                for i, result in enumerate(vector_results[:5]):  # Limit vector results for prompt brevity
                    data = result['data']
                    similarity = result.get('similarity_score', 0.0)
                    frameworks = self._get_framework_mappings(
                        data.get('metric_name', ''), 
                        data.get('metric_category', '')
                    )
                    framework_text = f" | Frameworks: {'; '.join(frameworks)}" if frameworks else ""
                    text_preview = (data.get('text_chunk') or data.get('text') or '')[:200]
                    filename = data.get('filename', 'Unknown')
                    category = (data.get('metric_category') or 'context').title()
                    
                    context_parts.append(
                        f"[{len(context_parts)+1}] VECTOR: {category}: {text_preview}... "
                        f"(Similarity: {similarity:.3f} | Source: {filename}{framework_text})"
                    )
            
            # If we still have no context, use the fallback answer
            if not context_parts:
                return self._generate_fallback_answer(question, results)
            
            context = '\n'.join(context_parts)
            
            prompt = f"""
            You are analyzing comprehensive social value data to answer questions with accurate citations.
            
            Question: {question}
            
            Available data (including both aggregated summaries and individual metrics):
            {context}
            
            Instructions:
            1. Answer the question based ONLY on the provided data
            2. For aggregation questions, prioritize AGGREGATED results over individual metrics
            3. Include specific numbers, values, and units where available
            4. Cite sources using [1], [2], etc. format
            5. Include cell references and framework mappings when available
            6. Note verification status (✓ verified, ⚠ failed, ? pending)
            7. If multiple data points contribute to an answer, sum them appropriately
            8. If data is insufficient, state that clearly
            9. Be comprehensive but concise
            
            Format your answer as:
            [Answer with specific data and calculations, citing all relevant sources]
            
            Sources with Details:
            [For each source, include: filename, cell references, and framework mappings when available]
            """
            
            response = self.openai_client.chat.completions.create(
                model=self.config.query_processing.gpt4_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.query_processing.gpt4_temperature,
                max_tokens=self.config.query_processing.gpt4_max_tokens
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
        except Exception as e:
            logger.error(f"Error generating GPT answer: {e}")
            return self._generate_fallback_answer(question, results)
    
    def _generate_fallback_answer(self, question: str, results: List[Dict[str, Any]]) -> str:
        """Generate comprehensive fallback answer when GPT-4 is not available."""
        if not results:
            return (
                "I couldn't find specific data to answer your question. "
                "This might be because:\n"
                "1. No relevant data has been ingested yet\n"
                "2. The question doesn't match available metrics\n"
                "3. Try rephrasing your question or ingest more data files"
            )
        
        answer_parts = [f"Based on the comprehensive data analysis, here's what I found:\n"]
        
        # Group and summarize results
        aggregated_results = [r for r in results if r['type'] == 'sql_aggregated_metric']
        individual_results = [r for r in results if r['type'] in ['sql_individual_metric', 'sql_metric']]
        faiss_results = [r for r in results if r['type'] == 'faiss_vector_match']
        
        # Show aggregated results first
        if aggregated_results:
            answer_parts.append("AGGREGATED TOTALS:")
            for i, result in enumerate(aggregated_results[:5]):
                data = result['data']
                frameworks = self._get_framework_mappings(
                    data.get('metric_name', ''), 
                    data.get('metric_category', '')
                )
                framework_text = f" | Frameworks: {'; '.join(frameworks)}" if frameworks else ""
                
                answer_parts.append(
                    f"{i+1}. {data['metric_category'].title()}: "
                    f"Total {data['metric_name']} = {data['total_value']} {data['metric_unit']} "
                    f"(from {data['count']} records){framework_text}"
                )
        
        # Show individual results
        if individual_results:
            answer_parts.append(f"\nINDIVIDUAL METRICS (showing {min(10, len(individual_results))} of {len(individual_results)}):")
            for i, result in enumerate(individual_results[:10]):
                data = result['data']
                frameworks = self._get_framework_mappings(
                    data.get('metric_name', ''), 
                    data.get('metric_category', '')
                )
                framework_text = f" | Frameworks: {'; '.join(frameworks)}" if frameworks else ""
                cell_ref = self._format_cell_reference(data)
                
                verification_status = data.get('verification_status', 'pending')
                status_indicator = "✓" if verification_status == 'verified' else "⚠" if verification_status == 'failed' else "?"
                
                answer_parts.append(
                    f"{i+1}. {data['metric_category'].title()}: "
                    f"{data['metric_name']} = {data['metric_value']} {data['metric_unit']} {status_indicator} "
                    f"({cell_ref}){framework_text}"
                )
        
        # Add FAISS vector results if significant
        if faiss_results:
            answer_parts.append(f"\nVECTOR SEARCH MATCHES (showing {min(5, len(faiss_results))} of {len(faiss_results)}):")
            for i, result in enumerate(faiss_results[:5]):
                data = result['data']
                similarity = result.get('similarity_score', 0)
                frameworks = self._get_framework_mappings(
                    data.get('metric_name', ''), 
                    data.get('metric_category', '')
                )
                framework_text = f" | Frameworks: {'; '.join(frameworks)}" if frameworks else ""
                
                answer_parts.append(
                    f"{i+1}. {data['metric_category'].title()}: "
                    f"{data['text_chunk'][:100]}... "
                    f"(Similarity: {similarity:.3f}){framework_text}"
                )
        
        # Add comprehensive source listing
        sources = []
        for result in results[:15]:
            data = result['data']
            filename = data.get('filename', 'Unknown')
            if filename not in [s.split(' (')[0] for s in sources]:
                cell_ref = self._format_cell_reference(data) if result['type'] not in ['vector_match', 'faiss_vector_match'] else ""
                if cell_ref and cell_ref != "No specific cell reference":
                    sources.append(f"{filename} ({cell_ref})")
                else:
                    sources.append(filename)
        
        if sources:
            answer_parts.append(f"\nSources with Details: {'; '.join(sources)}")
        
        answer_parts.append(f"\nTotal Results Analyzed: {len(results)}")
        
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