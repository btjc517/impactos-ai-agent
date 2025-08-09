"""
Query and Q&A system for ImpactOS AI Layer MVP Phase One.

This module handles natural language queries with enhanced functionality:
1. Replacing hard top-10 limit with relevance threshold filtering
2. Query-type specific result limits (higher for aggregations)
3. Two-stage processing: retrieve all relevant, then filter for GPT-5
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
from llm_utils import (
    choose_model,
    call_chat_completion,
    should_escalate_answer,
)

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
        self._intent_cache: Dict[str, Dict[str, Any]] = {}
        self._answer_cache: Dict[str, str] = {}
    
    def _initialize_openai(self) -> Optional[OpenAI]:
        """Initialize OpenAI client with API key."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.warning("OPENAI_API_KEY not found. GPT-5 orchestration disabled.")
            return None
        
        try:
            client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            return None
    
    def _build_answer_cache_key(self, question: str, results: List[Dict[str, Any]], model_name: Optional[str] = None) -> str:
        """Create a stable cache key from question, top source filenames, and model."""
        try:
            filenames = []
            for r in results[:10]:
                data = r.get('data', {})
                fn = data.get('filename') or data.get('filenames') or ''
                if isinstance(fn, str):
                    filenames.append(fn)
            parts = [question.strip().lower(), '|'.join(sorted(set(filenames)))]
            if model_name:
                parts.append(model_name)
            key = '||'.join(parts)
            return key
        except Exception:
            return question.strip().lower() if not model_name else f"{question.strip().lower()}||{model_name}"

    def _try_sql_direct_answer(self, question: str, analysis: Dict[str, Any], results: List[Dict[str, Any]]) -> Optional[str]:
        """Attempt to generate a deterministic aggregation answer without GPT.
        Returns answer string if confident, else None.
        """
        try:
            if os.getenv('IMPACTOS_DISABLE_SQL_DIRECT', '').lower() in ('1','true','yes'):
                return None
            ql = question.lower()
            sum_like = any(k in ql for k in ['total', 'sum', 'overall', 'aggregate'])
            if analysis.get('query_type') != 'aggregation' and 'sum' not in analysis.get('aggregations', []) and not sum_like:
                return None
            aggregated = [r for r in results if r['type'] == 'sql_aggregated_metric']
            individual = [r for r in results if r['type'] in ['sql_individual_metric', 'sql_metric']]
            # Prefer aggregated summaries when available
            if aggregated:
                # Ensure unit consistency
                units = {a['data'].get('metric_unit') for a in aggregated if a.get('data')}
                if len(units) != 1:
                    return None
                unit = next(iter(units)) or ''
                # If multiple metric_names present, bail out (ambiguous)
                metric_names = {a['data'].get('metric_name') for a in aggregated}
                if len(metric_names) != 1:
                    return None
                metric_name = next(iter(metric_names)) or 'metric'
                # Sum totals (aggregated rows are already SUM per group; if multiple groups exist, sum them)
                total = 0.0
                for a in aggregated:
                    val = a['data'].get('total_value')
                    try:
                        total += float(val)
                    except (TypeError, ValueError):
                        return None
                # Sources
                source_set = []
                for a in aggregated:
                    fns = a['data'].get('filenames') or ''
                    for fn in str(fns).split(','):
                        fn = fn.strip()
                        if fn and fn not in source_set:
                            source_set.append(fn)
                sources_fmt = ''.join([f"[{i+1}] {fn}\n" for i, fn in enumerate(source_set[:5])])
                answer = (
                    f"Total {metric_name.replace('_', ' ')} = {total:.2f} {unit}. "
                    + ' '.join([f"[{i+1}]" for i in range(min(5, len(source_set)))])
                )
                if source_set:
                    answer += f"\nSources:\n{sources_fmt}".rstrip()
                return answer
            # Fallback: compute sum from individual results if consistent
            if individual:
                units = {i['data'].get('metric_unit') for i in individual if i.get('data')}
                if len(units) != 1:
                    return None
                unit = next(iter(units)) or ''
                metric_names = {i['data'].get('metric_name') for i in individual}
                if len(metric_names) != 1:
                    return None
                metric_name = next(iter(metric_names)) or 'metric'
                total = 0.0
                for i in individual:
                    try:
                        total += float(i['data'].get('metric_value'))
                    except (TypeError, ValueError):
                        return None
                filenames = []
                for i in individual[:10]:
                    fn = i['data'].get('filename')
                    if fn and fn not in filenames:
                        filenames.append(fn)
                sources_fmt = ''.join([f"[{i+1}] {fn}\n" for i, fn in enumerate(filenames[:5])])
                answer = (
                    f"Total {metric_name.replace('_', ' ')} = {total:.2f} {unit}. "
                    + ' '.join([f"[{i+1}]" for i in range(min(5, len(filenames)))])
                )
                if filenames:
                    answer += f"\nSources:\n{sources_fmt}".rstrip()
                return answer
            return None
        except Exception as e:
            logger.warning(f"SQL direct answer attempt failed: {e}")
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
            
            # Answer cache check (quick return on repeats)
            # Build a provisional key using question only; later include sources
            provisional_key = question.strip().lower()
            if provisional_key in self._answer_cache:
                return self._answer_cache[provisional_key]
            
            # 1. Analyze query intent and extract key terms
            query_analysis = self._analyze_query(question)
            
            # 2. Perform enhanced hybrid search with intelligent limits
            results = self._enhanced_hybrid_search(question, query_analysis)
            
            logger.info(f"Retrieved {len(results)} total relevant results")
            
            # 2.5 SQL-first deterministic answer for aggregations
            direct_answer = self._try_sql_direct_answer(question, query_analysis, results)
            if direct_answer:
                cache_key = self._build_answer_cache_key(question, results, getattr(self, '_last_answer_model', None))
                self._answer_cache[cache_key] = direct_answer
                self._answer_cache[provisional_key] = direct_answer
                logger.info("Returning SQL-first deterministic answer (no GPT)")
                return direct_answer
            
            # 3. Intelligent filtering and summarization for GPT-5
            filtered_results = self._intelligent_filter_for_gpt(results, query_analysis)
            
            logger.info(f"Filtered to {len(filtered_results)} results for GPT-5")
            
            # 4. Generate cited answer using GPT-5 (if available)
            if self.openai_client and filtered_results:
                answer = self._generate_gpt_answer(question, filtered_results)
            else:
                answer = self._generate_fallback_answer(question, results)
            
            # Store in cache with stronger key after retrieval
            cache_key = self._build_answer_cache_key(question, results, getattr(self, '_last_answer_model', None))
            self._answer_cache[cache_key] = answer
            self._answer_cache[provisional_key] = answer
            logger.info("Query processing completed")
            return answer
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Error processing query: {e}"
    
    def _classify_intent_with_llm(self, question: str) -> Optional[Dict[str, Any]]:
        """Use a lightweight LLM to classify categories and aggregations.
        Returns a dict with keys: categories (list), aggregations (list), query_type (str).
        """
        cfg = self.config.analysis
        if not (self.openai_client and getattr(cfg, 'use_llm_for_intent', False)):
            return None
        if question in self._intent_cache:
            return self._intent_cache[question]
        try:
            categories = list(cfg.category_descriptions.keys()) if cfg.category_descriptions else []
            aggregations = list(cfg.aggregation_descriptions.keys()) if cfg.aggregation_descriptions else []
            prompt = (
                "Classify the question into zero or more categories and aggregations.\n"
                f"Categories: {', '.join(categories)}\n"
                f"Aggregations: {', '.join(aggregations)}\n"
                "Also classify overall query_type as one of: aggregation, descriptive, analytical.\n"
                "Return strict JSON with keys: categories (list of strings), aggregations (list of strings), query_type (string).\n"
                f"Question: {question}"
            )
            # Use centralized GPT-5 params with JSON enforcement and escalation
            from llm_utils import choose_model, call_chat_completion
            p = choose_model('intent')
            content, meta = call_chat_completion(
                self.openai_client,
                messages=[{"role": "user", "content": prompt}],
                model=p['model'],
                max_tokens=p['max_tokens'],
                reasoning=p.get('reasoning'),
                text=p.get('text'),
                enforce_json=True,
            )
            import json as _json
            data = None
            try:
                data = _json.loads(content) if content else None
            except Exception:
                data = None
            # Escalate if low confidence or malformed
            def _needs_escalation(d):
                if not d:
                    return True
                cats = d.get('categories') or []
                aggs = d.get('aggregations') or []
                qtype = d.get('query_type')
                if qtype not in ('aggregation','descriptive','analytical'):
                    return True
                # proxy confidence check: require at least one label
                return not (cats or aggs)
            if getattr(self.config.llm, 'escalation_enabled', True) and _needs_escalation(data):
                # escalate to mini
                content2, _ = call_chat_completion(
                    self.openai_client,
                    messages=[{"role": "user", "content": prompt}],
                    model='gpt-5-mini',
                    max_tokens=p['max_tokens'],
                    reasoning={'effort': 'low'},
                    text=p.get('text'),
                    enforce_json=True,
                )
                try:
                    d2 = _json.loads(content2) if content2 else None
                except Exception:
                    d2 = None
                data = d2 or data
                if _needs_escalation(data):
                    content3, _ = call_chat_completion(
                        self.openai_client,
                        messages=[{"role": "user", "content": prompt}],
                        model='gpt-5',
                        max_tokens=p['max_tokens'],
                        reasoning={'effort': 'low'},
                        text=p.get('text'),
                        enforce_json=True,
                    )
                    try:
                        d3 = _json.loads(content3) if content3 else None
                    except Exception:
                        d3 = None
                    data = d3 or data
            if isinstance(data, dict):
                self._intent_cache[question] = data
                return data
        except Exception as e:
            logger.warning(f"LLM intent classification failed: {e}")
        return None
    
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
        
        # Optionally classify with a lightweight LLM first
        llm_intent = self._classify_intent_with_llm(question)
        if llm_intent:
            analysis['categories'] = llm_intent.get('categories', [])
            analysis['aggregations'] = llm_intent.get('aggregations', [])
            if llm_intent.get('query_type') in ('aggregation','descriptive','analytical'):
                analysis['query_type'] = llm_intent['query_type']
        
        # Prefer embedding-based intent detection if available
        cfg = getattr(self.config, 'analysis', None)
        used_embedding_intent = False
        if cfg and getattr(self.config.analysis, 'use_embedding_for_intent', False) and self.vector_search and self.vector_search.embedding_model:
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
                        if label not in analysis['categories']:
                            analysis['categories'].append(label)
                
                # Aggregations
                for label, proto_vec in zip(agg_labels, agg_vecs):
                    sim = cos_sim(query_vec, proto_vec)
                    if sim >= cfg.aggregation_min_similarity:
                        if label not in analysis['aggregations']:
                            analysis['aggregations'].append(label)
                    used_embedding_intent = True
            except Exception as e:
                logger.warning(f"Embedding-based intent detection failed, using keywords as complement: {e}")
                used_embedding_intent = False
        
        # Always complement with keyword matching (union), to improve recall
        category_keywords = getattr(self.config.analysis, 'category_keywords', {})
        for category, keywords in category_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                if category not in analysis['categories']:
                    analysis['categories'].append(category)
        aggregation_patterns = getattr(self.config.analysis, 'aggregation_patterns', {})
        for agg_type, patterns in aggregation_patterns.items():
            if any(pattern in question_lower for pattern in patterns):
                if agg_type not in analysis['aggregations']:
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
        """Intelligently filter results for GPT-5 context window."""
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
        """Generate answer using GPT-5 with enhanced context."""
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
            
            # Detect if a graph/visualization was requested to guide tone
            ql = question.lower()
            graph_requested = any(k in ql for k in ["graph", "chart", "plot", "visual", "visualize", "visualisation", "visualization"])            
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
            10. Do NOT say that a graph cannot be provided; assume the UI will render any requested charts.
            11. If a chart is requested, include a short 1-line caption and recommended chart type (e.g., Bar or Line) and axes labels.
            
            Format your answer as:
            [Answer with specific data and calculations, citing all relevant sources]
            
            Sources with Details:
            [For each source, include: filename, cell references, and framework mappings when available]
            """
            
            base_params = choose_model('answer')
            content, meta = call_chat_completion(
                self.openai_client,
                messages=[{"role": "user", "content": prompt}],
                model=base_params['model'],
                max_tokens=base_params['max_tokens'],
                reasoning=base_params.get('reasoning'),
                text=base_params.get('text'),
                enforce_json=False,
            )
            answer = (content or '').strip()
            used_model = base_params['model']

            # Check guardrails for escalation
            escalate = False
            reason = ''
            if getattr(self.config.llm, 'escalation_enabled', True):
                escalate, reason = should_escalate_answer(
                    answer_text=answer,
                    context_count=len(context_parts),
                    latency_ms=meta.get('latency_ms', 0),
                    qa_cfg=self.config.qa,
                )
            if escalate:
                logger.info(f"Escalating answer synthesis due to: {reason}")
                high_params = base_params.copy()
                high_params['model'] = 'gpt-5'
                high_params['max_tokens'] = max(4000, base_params.get('max_tokens', 2000))
                high_params['reasoning'] = {'effort': 'high'}
                content2, meta2 = call_chat_completion(
                    self.openai_client,
                    messages=[{"role": "user", "content": prompt}],
                    model=high_params['model'],
                    max_tokens=high_params['max_tokens'],
                    reasoning=high_params.get('reasoning'),
                    text=high_params.get('text'),
                    enforce_json=False,
                )
                if content2:
                    ans2 = content2.strip()
                    esc_again, _ = should_escalate_answer(
                        ans2, len(context_parts), meta2.get('latency_ms', 0), self.config.qa
                    )
                    if not esc_again:
                        answer = ans2
                        used_model = high_params['model']
            # Sanitize any residual "cannot provide graph" disclaimers if graph requested
            if graph_requested:
                answer = self._sanitize_graph_language(answer)
            self._last_answer_model = used_model
            return answer
        except Exception as e:
            logger.error(f"Error generating GPT answer: {e}")
            return self._generate_fallback_answer(question, results)

    def _sanitize_graph_language(self, text: str) -> str:
        """Remove unhelpful disclaimers about inability to display graphs when UI will render charts."""
        try:
            lowered = text.lower()
            bad_phrases = [
                "cannot be provided in this text-based format",
                "cannot be provided in a text-based format",
                "cannot provide a graph",
                "unable to provide a graph",
                "cannot display a chart",
                "cannot display the graph",
                "i cannot provide",
                "this text-based format"
            ]
            cleaned_lines = []
            for line in text.split('\n'):
                lline = line.lower()
                if any(p in lline for p in bad_phrases):
                    continue
                cleaned_lines.append(line)
            return '\n'.join(cleaned_lines)
        except Exception:
            return text
    
    def _generate_fallback_answer(self, question: str, results: List[Dict[str, Any]]) -> str:
        """Generate comprehensive fallback answer when GPT-5 is not available."""
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

    # =====================
    # Visualization support
    # =====================
    def _detect_chart_intent(self, question: str, analysis: Dict[str, Any]) -> Tuple[bool, str]:
        """Detect if the user likely wants a chart and suggest a type.
        Returns (want_chart, chart_type) where chart_type is 'bar' or 'line'.
        """
        q = question.lower()
        chart_keywords = ["chart", "graph", "plot", "visual", "visualize", "visualisation", "visualization", "bar", "line", "trend", "over time", "timeseries", "time series", "distribution", "breakdown"]
        want_chart = any(k in q for k in chart_keywords)
        # Prefer a chart for aggregation results even if not explicitly asked
        if analysis.get('query_type') == 'aggregation':
            want_chart = True or want_chart
        # Choose type
        time_like = any(k in q for k in ["over time", "trend", "timeseries", "time series", "monthly", "quarterly", "annually", "per month", "per quarter", "per year"]) or any(t in analysis.get('time_references', []) for t in ["monthly", "quarterly", "annually"])
        chart_type = 'line' if time_like else 'bar'
        return want_chart, chart_type

    def _is_time_series_request(self, question: str, analysis: Dict[str, Any]) -> bool:
        q = question.lower()
        return any(k in q for k in ["over time", "trend", "timeseries", "time series", "monthly", "quarterly", "annually", "per month", "per quarter", "per year", "by month", "by year", "timeline"]) or analysis.get('query_type') == 'aggregation' and any(t in (analysis.get('time_references') or []) for t in ["monthly", "quarterly", "annually"])

    def _build_time_series_chart(self, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build a time-series chart payload by aggregating metric values by month.
        Uses im.timestamp when available, otherwise falls back to source processed_timestamp.
        Returns payload with type 'line', x_key 'date', multi-series data by category or metric_name.
        """
        try:
            import sqlite3
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                where_conditions = ["im.metric_value IS NOT NULL"]
                params: List[Any] = []
                categories = analysis.get('categories') or []
                if categories:
                    placeholders = ','.join(['?' for _ in categories])
                    where_conditions.append(f"im.metric_category IN ({placeholders})")
                    params.extend(categories)
                where_clause = ' AND '.join(where_conditions)
                # Aggregate by month using im.timestamp or fallback to source processed timestamp
                query = f"""
                    SELECT 
                        COALESCE(strftime('%Y-%m', im.timestamp), strftime('%Y-%m', s.processed_timestamp)) AS period,
                        im.metric_category,
                        im.metric_name,
                        SUM(im.metric_value) AS total_value
                    FROM impact_metrics im
                    JOIN sources s ON im.source_id = s.id
                    WHERE {where_clause}
                    GROUP BY period, im.metric_category, im.metric_name
                    HAVING period IS NOT NULL
                    ORDER BY period ASC
                """
                cursor.execute(query, params)
                rows = [dict(r) for r in cursor.fetchall()]
                if not rows:
                    return None
                # Decide series dimension: prefer categories if present, else top metric_names
                series_by = 'metric_category' if categories else 'metric_name'
                # Collect periods
                periods = sorted({r['period'] for r in rows})
                # Choose up to 4 series keys by total sum
                totals: Dict[str, float] = {}
                for r in rows:
                    key = r[series_by] or 'metric'
                    totals[key] = totals.get(key, 0.0) + float(r['total_value'] or 0.0)
                top_keys = [k for k, _ in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:4]]
                # Build data rows with zero-fill
                data_rows: List[Dict[str, Any]] = []
                for p in periods:
                    row: Dict[str, Any] = {'date': p}
                    for key in top_keys:
                        row[key] = 0.0
                    for r in [rr for rr in rows if rr['period'] == p]:
                        key = (r[series_by] or 'metric')
                        if key in top_keys:
                            row[key] += float(r['total_value'] or 0.0)
                    data_rows.append(row)
                # Build series descriptors and config colors
                palette = [
                    'hsl(var(--chart-1))',
                    'hsl(var(--chart-2))',
                    'hsl(var(--chart-3))',
                    'hsl(var(--chart-4))',
                ]
                series = []
                config = {}
                for i, key in enumerate(top_keys):
                    series.append({'key': key, 'label': key.replace('_', ' ').title(), 'color': palette[i % len(palette)]})
                    config[key] = {'label': key.replace('_', ' ').title(), 'color': palette[i % len(palette)]}
                return {
                    'type': 'line',
                    'x_key': 'date',
                    'series': series,
                    'data': data_rows,
                    'config': config,
                    'meta': {
                        'grouped_by': series_by,
                        'period': 'month'
                    }
                }
        except Exception as e:
            logger.debug(f"Failed to build time-series chart: {e}")
            return None

    def _build_chart_data(self, results: List[Dict[str, Any]], analysis: Dict[str, Any], chart_type: str) -> Optional[Dict[str, Any]]:
        """Construct a shadcn/Recharts-friendly chart payload from results.
        Returns a dict with keys: type, x_key, series, data, config, meta.
        """
        try:
            # Prefer aggregated SQL results
            aggregated = [r for r in results if r['type'] == 'sql_aggregated_metric']
            individual = [r for r in results if r['type'] in ['sql_individual_metric', 'sql_metric']]
            vector_matches = [r for r in results if r['type'] == 'faiss_vector_match']
            
            data_rows: List[Dict[str, Any]] = []
            unit = None
            metric_name = None
            
            if aggregated:
                # Build a simple bar dataset: label = metric_name (with category), value = total_value
                for row in aggregated:
                    d = row['data']
                    if unit is None:
                        unit = d.get('metric_unit')
                    if metric_name is None:
                        metric_name = d.get('metric_name')
                    label = f"{d.get('metric_name','')}" if d.get('metric_name') else (d.get('metric_category') or 'metric')
                    data_rows.append({
                        'label': label,
                        'value': float(d.get('total_value') or 0),
                    })
            elif individual:
                # Fallback: sum values by metric_name
                sums: Dict[str, float] = {}
                units: Dict[str, str] = {}
                for r in individual:
                    d = r['data']
                    name = d.get('metric_name') or 'metric'
                    try:
                        v = float(d.get('metric_value'))
                    except (TypeError, ValueError):
                        continue
                    sums[name] = sums.get(name, 0.0) + v
                    if name not in units and d.get('metric_unit'):
                        units[name] = d.get('metric_unit')
                # Take top 10
                for name, total in sorted(sums.items(), key=lambda kv: kv[1], reverse=True)[:10]:
                    data_rows.append({'label': name, 'value': float(total)})
                unit = next(iter(units.values()), None)
                metric_name = None
            elif vector_matches:
                # Fallback: use FAISS vector results and aggregate by metric_name using metric_value from metadata
                sums: Dict[str, float] = {}
                units: Dict[str, str] = {}
                for r in vector_matches:
                    d = r.get('data', {})
                    name = d.get('metric_name') or 'metric'
                    # Value may be directly present or under source_info from vector_search
                    val = d.get('metric_value')
                    if val is None:
                        val = d.get('value')
                    if val is None:
                        # source_info fields were merged; try both
                        val = d.get('source_info.metric_value')
                    try:
                        v = float(val)
                    except (TypeError, ValueError):
                        continue
                    sums[name] = sums.get(name, 0.0) + v
                    unit_val = d.get('metric_unit') or d.get('unit')
                    if name not in units and unit_val:
                        units[name] = unit_val
                for name, total in sorted(sums.items(), key=lambda kv: kv[1], reverse=True)[:10]:
                    data_rows.append({'label': name, 'value': float(total)})
                unit = next(iter(units.values()), None)
                metric_name = None
            else:
                return None
            
            if not data_rows:
                return None
            
            # Build ChartConfig compatible with shadcn/ui
            chart_payload: Dict[str, Any] = {
                'type': chart_type,
                'x_key': 'label',
                'series': [
                    {
                        'key': 'value',
                        'label': 'Total',
                        'color': 'hsl(var(--chart-1))'
                    }
                ],
                'data': data_rows,
                'config': {
                    'value': {
                        'label': 'Total',
                        'color': 'hsl(var(--chart-1))'
                    }
                },
                'meta': {
                    'unit': unit,
                    'metric_name': metric_name,
                }
            }
            return chart_payload
        except Exception as e:
            logger.debug(f"Failed to build chart data: {e}")
            return None

    def query_structured(self, question: str, force_chart: Optional[bool] = None) -> Dict[str, Any]:
        """Process query and return a structured response with optional chart data.
        Keys: answer (str), show_chart (bool), chart (dict or None).
        """
        logger.info(f"Processing structured query: {question}")
        # Analyze and retrieve results using existing pipeline
        analysis = self._analyze_query(question)
        results = self._enhanced_hybrid_search(question, analysis)
        
        # Try deterministic answer first
        answer = self._try_sql_direct_answer(question, analysis, results)
        if not answer:
            filtered_results = self._intelligent_filter_for_gpt(results, analysis)
            if self.openai_client and filtered_results:
                answer = self._generate_gpt_answer(question, filtered_results)
            else:
                answer = self._generate_fallback_answer(question, results)
        
        # Visualization intent and payload
        want_chart_default, suggested_type = self._detect_chart_intent(question, analysis)
        want_chart = force_chart if force_chart is not None else want_chart_default
        chart_payload = None
        if want_chart:
            # Prefer time-series chart when requested
            if suggested_type == 'line' or self._is_time_series_request(question, analysis):
                chart_payload = self._build_time_series_chart(analysis)
            # Fallback to categorical chart
            if chart_payload is None:
                chart_payload = self._build_chart_data(results, analysis, suggested_type)
            # If we wanted a chart but couldn't build one, disable
            if chart_payload is None:
                want_chart = False
        
        return {
            'answer': answer,
            'show_chart': bool(want_chart),
            'chart': chart_payload
        }


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