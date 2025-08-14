"""
Web API service for ImpactOS AI system integration.

This module provides a FastAPI web service that exposes the ImpactOS AI system
functionality through REST API endpoints for web portal integration.
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional
import time
from datetime import datetime
from pathlib import Path

# FastAPI and web service imports
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager

# Add src to path for imports
src_path = os.path.dirname(__file__)
if src_path not in sys.path:
    sys.path.append(src_path)

# Also add project root to path (for when running from project root)
project_root = os.path.dirname(src_path)
if project_root not in sys.path:
    sys.path.append(project_root)

# Local imports
from main import ImpactOSCLI
from query import QuerySystem, query_data
from telemetry import telemetry, capture_logs
from schema import DatabaseSchema
from frameworks import get_framework_report, apply_framework_mappings
from verify import verify_all_data, verify_metric
from config import get_config

# Testing infrastructure imports (with try/except for optional functionality)
try:
    import sys
    import os
    testing_path = os.path.join(os.path.dirname(__file__), 'testing')
    if testing_path not in sys.path:
        sys.path.append(testing_path)
    
    from testing.test_runner import TestRunner
    from testing.performance_tracker import PerformanceTracker
    from testing.test_cases import TestCases
    from testing.test_database import TestDatabase
    TESTING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Testing infrastructure not available: {e}")
    TESTING_AVAILABLE = False

# Additional imports
from vector_search import FAISSVectorSearch
import sqlite3
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question about the data")
    show_accuracy: bool = Field(False, description="Include verification accuracy in response")
    force_chart: Optional[bool] = Field(None, description="Force chart rendering if possible")
    timezone: Optional[str] = Field(None, description="Client timezone, e.g., Europe/Paris")
    # Optional identifiers passed from the web portal
    user_id: Optional[str] = Field(None, description="Authenticated user id from web portal auth")
    session_id: Optional[str] = Field(None, description="Client session id for grouping queries")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="Answer to the question with citations")
    accuracy_summary: Optional[str] = Field(None, description="Data accuracy summary if requested")
    timestamp: datetime = Field(default_factory=datetime.now)
    # New fields for frontend chart rendering
    show_chart: Optional[bool] = Field(False, description="Whether the frontend should render a chart for this answer")
    chart: Optional[Dict[str, Any]] = Field(None, description="Chart payload compatible with shadcn/Recharts: {type,x_key,series,data,config,meta}")
    time_window: Optional[Dict[str, Any]] = Field(None, description="Resolved time window for the query")

class IngestionRequest(BaseModel):
    file_path: str = Field(..., description="Path to file to ingest")
    file_type: Optional[str] = Field(None, description="File type override (auto-detected if not provided)")
    verify_after_ingestion: bool = Field(False, description="Run verification after ingestion")
    bronze_only: bool = Field(False, description="Only create Bronze tables and sheet registry (no metric extraction)")
    auto_transform: bool = Field(True, description="Enqueue Silver transforms when new sheet versions appear")
    mode: Optional[str] = Field(None, description="Transform mode: sync|async (defaults to env)")

class IngestionResponse(BaseModel):
    success: bool = Field(..., description="Whether ingestion was successful")
    message: str = Field(..., description="Success or error message")
    verification_results: Optional[Dict[str, Any]] = Field(None, description="Verification results if requested")
    timestamp: datetime = Field(default_factory=datetime.now)

class VerificationRequest(BaseModel):
    target: str = Field(..., description="Verification target: 'all' for all metrics, or specific metric ID")

class VerificationResponse(BaseModel):
    results: Dict[str, Any] = Field(..., description="Verification results")
    timestamp: datetime = Field(default_factory=datetime.now)

class SchemaResponse(BaseModel):
    schema_info: Dict[str, List[str]] = Field(..., description="Database schema information")
    database_path: str = Field(..., description="Database file path")
    timestamp: datetime = Field(default_factory=datetime.now)

class DataListResponse(BaseModel):
    available_files: List[Dict[str, Any]] = Field(..., description="Available data files")
    ingested_sources: List[Dict[str, Any]] = Field(..., description="Ingested data sources")
    timestamp: datetime = Field(default_factory=datetime.now)

class FrameworksResponse(BaseModel):
    report: str = Field(..., description="Framework mapping report")
    applied_mappings: Optional[int] = Field(None, description="Number of applied mappings if requested")
    timestamp: datetime = Field(default_factory=datetime.now)

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="API version")
    database_connected: bool = Field(..., description="Database connection status")
    openai_configured: bool = Field(..., description="OpenAI API configuration status")
    timestamp: datetime = Field(default_factory=datetime.now)

class TestRequest(BaseModel):
    test_types: Optional[List[str]] = Field(None, description="Types of tests to run (e.g., ['accuracy', 'performance'])")
    notes: Optional[str] = Field(None, description="Notes about this test run")

class TestResponse(BaseModel):
    results: Dict[str, Any] = Field(..., description="Test execution results")
    test_run_id: Optional[int] = Field(None, description="Test run ID for tracking")
    timestamp: datetime = Field(default_factory=datetime.now)

class PerformanceRequest(BaseModel):
    days: int = Field(30, description="Number of days to analyze")
    environment: Optional[str] = Field(None, description="Environment filter")

class PerformanceResponse(BaseModel):
    report: Dict[str, Any] = Field(..., description="Performance analysis report")
    timestamp: datetime = Field(default_factory=datetime.now)

class ConfigUpdateRequest(BaseModel):
    config_section: str = Field(..., description="Configuration section to update")
    config_data: Dict[str, Any] = Field(..., description="Configuration data to update")

class ConfigResponse(BaseModel):
    current_config: Dict[str, Any] = Field(..., description="Current system configuration")
    updated: bool = Field(False, description="Whether configuration was updated")
    timestamp: datetime = Field(default_factory=datetime.now)

class DataAnalysisRequest(BaseModel):
    analysis_type: str = Field(..., description="Type of analysis: 'metrics_summary', 'framework_coverage', 'data_quality'")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters for analysis")

class DataAnalysisResponse(BaseModel):
    analysis_type: str = Field(..., description="Type of analysis performed")
    results: Dict[str, Any] = Field(..., description="Analysis results")
    timestamp: datetime = Field(default_factory=datetime.now)

class VectorSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(10, description="Maximum number of results")
    min_similarity: Optional[float] = Field(None, description="Minimum similarity threshold")

class VectorSearchResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(..., description="Vector search results")
    query: str = Field(..., description="Original query")
    total_found: int = Field(..., description="Total number of results found")
    timestamp: datetime = Field(default_factory=datetime.now)

# Initialize ImpactOS CLI instance
impactos_cli = None

def get_cli_instance():
    """Get or create ImpactOS CLI instance."""
    global impactos_cli
    if impactos_cli is None:
        impactos_cli = ImpactOSCLI()
    return impactos_cli

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Startup
    logger.info("Starting ImpactOS AI Web API service...")
    cli = get_cli_instance()
    logger.info("ImpactOS AI Web API service ready")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ImpactOS AI Web API service...")
    # Cleanup if needed

# Initialize FastAPI app
app = FastAPI(
    title="ImpactOS AI API",
    description="REST API for ImpactOS AI social value data analysis system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware for web portal integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now - configure for production
    allow_credentials=False,  # Set to False when using allow_origins=["*"]
    allow_methods=["GET", "POST", "HEAD", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.get("/", response_model=Dict[str, str])
@app.head("/")
async def root():
    """API root endpoint."""
    return {
        "message": "ImpactOS AI Web API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        cli = get_cli_instance()
        
        # Check database connection
        db_connected = os.path.exists(cli.db_path)
        
        # Check OpenAI configuration
        openai_configured = bool(os.getenv('OPENAI_API_KEY'))
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            database_connected=db_connected,
            openai_configured=openai_configured
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint for debugging."""
    return {
        "status": "API is working",
        "timestamp": datetime.now().isoformat(),
        "environment": "production" if os.getenv("PORT") else "development"
    }

@app.post("/query", response_model=QueryResponse)
async def query_data_endpoint(request: QueryRequest):
    """
    Query the system with natural language question.
    
    This is the main endpoint your web portal query bar should use.
    """
    try:
        cli = get_cli_instance()
        
        logger.info(f"Processing web API query: {request.question}")
        
        # Process the query using structured response (answer + optional chart) with instrumentation
        qs = QuerySystem(cli.db_path)
        started = time.monotonic()
        with capture_logs() as log_handler:
            # Accept timezone override via request, else rely on env/default inside QuerySystem
            if request.timezone:
                os.environ['TIMEZONE'] = request.timezone
            structured, timings, model_used = qs.query_structured_instrumented(
                request.question,
                force_chart=request.force_chart,
            )
        answer = structured.get('answer', '')
        
        # Get accuracy summary if requested
        accuracy_summary = None
        if request.show_accuracy:
            accuracy_summary = cli.get_verification_summary()
        
        response = QueryResponse(
            answer=answer,
            accuracy_summary=accuracy_summary,
            show_chart=structured.get('show_chart', False),
            chart=structured.get('chart'),
            time_window=structured.get('time_window')
        )
        # Fire-and-forget telemetry (synchronous call with small timeout; non-blocking failure)
        try:
            if telemetry.is_enabled():
                total_ms = timings.get('total_ms') if isinstance(timings, dict) else int((time.monotonic() - started) * 1000)
                logs_text = log_handler.get_value()
                event = telemetry.build_event(
                    question=request.question,
                    answer=answer,
                    status='ok',
                    source='web',
                    user_id=request.user_id,
                    session_id=request.session_id,
                    model=model_used,
                    total_ms=total_ms,
                    timings=timings if isinstance(timings, dict) else {},
                    chart=structured.get('chart'),
                    logs_text=logs_text,
                    metadata={'show_accuracy': request.show_accuracy, 'force_chart': request.force_chart},
                )
                telemetry.send_query_event(event)
        except Exception:
            pass
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        # Attempt to send a failed telemetry event as well
        try:
            if telemetry.is_enabled():
                event = telemetry.build_event(
                    question=getattr(request, 'question', ''),
                    answer=None,
                    status='error',
                    source='web',
                    user_id=getattr(request, 'user_id', None),
                    session_id=getattr(request, 'session_id', None),
                    model=None,
                    total_ms=None,
                    timings=None,
                    chart=None,
                    logs_text=None,
                    error=str(e),
                    metadata={'endpoint': '/query'},
                )
                telemetry.send_query_event(event)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/ingest", response_model=IngestionResponse)
async def ingest_data_endpoint(request: IngestionRequest):
    """Ingest data from file into the system."""
    try:
        cli = get_cli_instance()
        
        logger.info(f"Processing ingestion request: {request.file_path}")
        
        # Check if file exists
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        
        # Perform ingestion
        if request.bronze_only:
            from bronze_ingest import ingest_bronze
            if request.auto_transform:
                if request.mode:
                    os.environ['TRANSFORM_MODE'] = request.mode
                os.environ['AUTO_TRANSFORM'] = 'true'
            res = ingest_bronze(request.file_path, cli.db_path)
            success = True
        else:
            success = cli.ingest_data(request.file_path, request.file_type)
        
        # Run verification if requested
        verification_results = None
        if request.verify_after_ingestion and success:
            verification_results = verify_all_data(cli.db_path)
        
        msg = "Bronze ingest completed" if request.bronze_only and success else ("File ingested successfully" if success else "Ingestion failed")
        return IngestionResponse(success=success, message=msg, verification_results=verification_results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/upload", response_model=IngestionResponse)
async def upload_and_ingest(
    file: UploadFile = File(...),
    verify_after_ingestion: bool = Query(False, description="Run verification after ingestion")
):
    """Upload and ingest a file in one step."""
    try:
        # Create uploads directory if it doesn't exist
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Save uploaded file
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"File uploaded: {file_path}")
        
        # Process ingestion
        cli = get_cli_instance()
        success = cli.ingest_data(str(file_path), None)
        
        # Run verification if requested
        verification_results = None
        if verify_after_ingestion and success:
            verification_results = verify_all_data(cli.db_path)
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return IngestionResponse(
            success=success,
            message=f"File {file.filename} processed successfully" if success else f"Processing {file.filename} failed",
            verification_results=verification_results
        )
        
    except Exception as e:
        logger.error(f"Error during file upload and ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"Upload and ingestion failed: {str(e)}")

@app.post("/verify", response_model=VerificationResponse)
async def verify_data_endpoint(request: VerificationRequest):
    """Verify data accuracy."""
    try:
        cli = get_cli_instance()
        
        logger.info(f"Processing verification request: {request.target}")
        
        if request.target.lower() == 'all':
            results = verify_all_data(cli.db_path)
        else:
            try:
                metric_id = int(request.target)
                results = verify_metric(metric_id, cli.db_path)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid metric ID: {request.target}")
        
        return VerificationResponse(results=results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")

@app.get("/schema", response_model=SchemaResponse)
async def get_schema_info():
    """Get database schema information."""
    try:
        cli = get_cli_instance()
        schema_info = cli.db_schema.get_schema_info()
        
        return SchemaResponse(
            schema_info=schema_info,
            database_path=cli.db_path
        )
        
    except Exception as e:
        logger.error(f"Error getting schema info: {e}")
        raise HTTPException(status_code=500, detail=f"Schema retrieval failed: {str(e)}")

@app.get("/data", response_model=DataListResponse)
async def list_data():
    """List available data files and ingested sources."""
    try:
        cli = get_cli_instance()
        
        # Get available files
        available_files = []
        data_dir = Path("data")
        if data_dir.exists():
            for file_path in sorted(data_dir.glob("*")):
                if file_path.is_file():
                    available_files.append({
                        "filename": file_path.name,
                        "size_bytes": file_path.stat().st_size,
                        "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    })
        
        # Get ingested sources from database
        ingested_sources = []
        try:
            import sqlite3
            with sqlite3.connect(cli.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT s.filename, s.processed_timestamp, s.processing_status,
                           COUNT(im.id) as metric_count,
                           AVG(im.verification_accuracy) as avg_accuracy
                    FROM sources s
                    LEFT JOIN impact_metrics im ON s.id = im.source_id
                    GROUP BY s.id
                    ORDER BY s.processed_timestamp DESC
                """)
                
                sources = cursor.fetchall()
                for source in sources:
                    ingested_sources.append({
                        "filename": source['filename'],
                        "processed_timestamp": source['processed_timestamp'],
                        "processing_status": source['processing_status'],
                        "metric_count": source['metric_count'],
                        "avg_accuracy": source['avg_accuracy'] or 0.0
                    })
        except Exception as e:
            logger.debug(f"Error querying sources: {e}")
        
        return DataListResponse(
            available_files=available_files,
            ingested_sources=ingested_sources
        )
        
    except Exception as e:
        logger.error(f"Error listing data: {e}")
        raise HTTPException(status_code=500, detail=f"Data listing failed: {str(e)}")

@app.get("/frameworks", response_model=FrameworksResponse)
async def get_frameworks_report(apply_mappings: bool = Query(False, description="Apply framework mappings to all metrics")):
    """Get framework mapping report and optionally apply mappings."""
    try:
        cli = get_cli_instance()
        
        applied_mappings = None
        if apply_mappings:
            applied_mappings = apply_framework_mappings(cli.db_path)
        
        report = get_framework_report(cli.db_path)
        
        return FrameworksResponse(
            report=report,
            applied_mappings=applied_mappings
        )
        
    except Exception as e:
        logger.error(f"Error generating framework report: {e}")
        raise HTTPException(status_code=500, detail=f"Framework report failed: {str(e)}")

@app.get("/config")
async def get_configuration():
    """Get current system configuration."""
    try:
        config = get_config()
        return {
            "vector_search": config.vector_search.__dict__,
            "query_processing": config.query_processing.__dict__,
            "extraction": config.extraction.__dict__,
            "scalability": config.scalability.__dict__
        }
    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration retrieval failed: {str(e)}")

@app.post("/config", response_model=ConfigResponse)
async def update_configuration(request: ConfigUpdateRequest):
    """Update system configuration."""
    try:
        current_config = get_config()
        
        # For now, return current config without actually updating
        # In production, you'd implement proper config management
        return ConfigResponse(
            current_config={
                "vector_search": current_config.vector_search.__dict__,
                "query_processing": current_config.query_processing.__dict__,
                "extraction": current_config.extraction.__dict__,
                "scalability": current_config.scalability.__dict__
            },
            updated=False  # Would be True after implementing config updates
        )
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration update failed: {str(e)}")

@app.post("/test", response_model=TestResponse)
async def run_tests(request: TestRequest):
    """Run comprehensive test suite."""
    if not TESTING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Testing infrastructure not available")
    
    try:
        logger.info(f"Starting test suite with types: {request.test_types}")
        
        test_runner = TestRunner()
        results = test_runner.run_comprehensive_test_suite(
            test_types=request.test_types,
            notes=request.notes
        )
        
        return TestResponse(
            results=results,
            test_run_id=results.get('test_run_id')
        )
        
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        raise HTTPException(status_code=500, detail=f"Test execution failed: {str(e)}")

@app.get("/test/performance", response_model=PerformanceResponse)
async def get_performance_report(request: PerformanceRequest = None):
    """Get performance analysis report."""
    if not TESTING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Testing infrastructure not available")
        
    try:
        days = request.days if request else 30
        logger.info(f"Generating performance report for {days} days")
        
        tracker = PerformanceTracker()
        report = tracker.generate_performance_report(days=days)
        
        return PerformanceResponse(report=report)
        
    except Exception as e:
        logger.error(f"Error generating performance report: {e}")
        raise HTTPException(status_code=500, detail=f"Performance report failed: {str(e)}")

@app.get("/test/history")
async def get_test_history(limit: int = Query(20, description="Maximum number of test runs to return")):
    """Get test execution history."""
    if not TESTING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Testing infrastructure not available")
        
    try:
        test_db = TestDatabase()
        history = test_db.get_test_history(limit=limit)
        
        return {
            "test_history": history,
            "total_runs": len(history),
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error getting test history: {e}")
        raise HTTPException(status_code=500, detail=f"Test history retrieval failed: {str(e)}")

@app.post("/search/vector", response_model=VectorSearchResponse)
async def vector_search(request: VectorSearchRequest):
    """Perform vector search on the database."""
    try:
        cli = get_cli_instance()
        
        # Initialize vector search
        vector_search = FAISSVectorSearch(cli.db_path)
        
        # Perform search
        results = vector_search.search(
            query=request.query,
            k=request.limit,
            min_similarity=request.min_similarity
        )
        
        # Format results for API response
        formatted_results = []
        for result in results:
            # Adapt to FAISSVectorSearch result structure
            data = result.get('data', {}) if isinstance(result, dict) else {}
            formatted_results.append({
                "text": data.get('text_chunk', ''),
                "similarity": result.get('similarity_score', 0.0),
                "source": data.get('filename', ''),
                "metric_id": data.get('metric_id'),
                "metadata": data
            })
        
        return VectorSearchResponse(
            results=formatted_results,
            query=request.query,
            total_found=len(formatted_results)
        )
        
    except Exception as e:
        logger.error(f"Error in vector search: {e}")
        raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")

@app.post("/search/rebuild-index")
async def rebuild_faiss_index():
    """Rebuild FAISS index from database embeddings without re-ingesting files."""
    try:
        cli = get_cli_instance()
        vector_search = FAISSVectorSearch(cli.db_path)
        vector_search.rebuild_index_from_database()
        stats = vector_search.get_stats() if hasattr(vector_search, 'get_stats') else {}
        return {"status": "ok", "stats": stats, "timestamp": datetime.now()}
    except Exception as e:
        logger.error(f"Error rebuilding FAISS index: {e}")
        raise HTTPException(status_code=500, detail=f"Rebuild index failed: {str(e)}")

@app.post("/analyze", response_model=DataAnalysisResponse)
async def analyze_data(request: DataAnalysisRequest):
    """Perform advanced data analysis."""
    try:
        cli = get_cli_instance()
        results = {}
        
        if request.analysis_type == "metrics_summary":
            results = _get_metrics_summary(cli.db_path, request.filters)
        elif request.analysis_type == "framework_coverage":
            results = _get_framework_coverage(cli.db_path, request.filters)
        elif request.analysis_type == "data_quality":
            results = _get_data_quality_analysis(cli.db_path, request.filters)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown analysis type: {request.analysis_type}")
        
        return DataAnalysisResponse(
            analysis_type=request.analysis_type,
            results=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in data analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Data analysis failed: {str(e)}")

@app.get("/metrics/live")
async def get_live_metrics():
    """Get real-time system metrics."""
    try:
        cli = get_cli_instance()
        
        # Database metrics
        db_stats = {}
        with sqlite3.connect(cli.db_path) as conn:
            cursor = conn.cursor()
            
            # Count metrics by type
            cursor.execute("SELECT COUNT(*) FROM impact_metrics")
            db_stats['total_metrics'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT source_id) FROM impact_metrics")
            db_stats['unique_sources'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(verification_accuracy) FROM impact_metrics WHERE verification_accuracy IS NOT NULL")
            result = cursor.fetchone()[0]
            db_stats['avg_accuracy'] = result if result else 0.0
            
            # Framework mapping stats
            cursor.execute("""
                SELECT framework_name, COUNT(*) 
                FROM framework_mappings 
                GROUP BY framework_name
            """)
            framework_stats = dict(cursor.fetchall())
            
        # Vector search stats
        vector_stats = {}
        try:
            vector_search = FAISSVectorSearch(cli.db_path)
            if hasattr(vector_search, 'index') and vector_search.index:
                vector_stats['total_vectors'] = vector_search.index.ntotal
            else:
                vector_stats['total_vectors'] = 0
        except:
            vector_stats['total_vectors'] = 0
        
        return {
            "database_stats": db_stats,
            "framework_stats": framework_stats,
            "vector_stats": vector_stats,
            "system_health": "healthy",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error getting live metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Live metrics failed: {str(e)}")

# Helper functions for data analysis
def _get_metrics_summary(db_path: str, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Get comprehensive metrics summary."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Basic counts
        cursor.execute("SELECT COUNT(*) as total FROM impact_metrics")
        total_metrics = cursor.fetchone()['total']
        
        cursor.execute("SELECT COUNT(DISTINCT source_id) as unique_sources FROM impact_metrics")
        unique_sources = cursor.fetchone()['unique_sources']
        
        # Accuracy distribution
        cursor.execute("""
            SELECT 
                AVG(verification_accuracy) as avg_accuracy,
                MIN(verification_accuracy) as min_accuracy,
                MAX(verification_accuracy) as max_accuracy
            FROM impact_metrics 
            WHERE verification_accuracy IS NOT NULL
        """)
        accuracy_stats = dict(cursor.fetchone())
        
        # Metric categories distribution
        cursor.execute("""
            SELECT metric_category, COUNT(*) as count
            FROM impact_metrics
            GROUP BY metric_category
            ORDER BY count DESC
        """)
        metric_types = [dict(row) for row in cursor.fetchall()]
        
        return {
            "total_metrics": total_metrics,
            "unique_sources": unique_sources,
            "accuracy_stats": accuracy_stats,
            "metric_types_distribution": metric_types
        }

def _get_framework_coverage(db_path: str, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Get framework mapping coverage analysis."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Framework mapping distribution
        cursor.execute("""
            SELECT 
                fm.framework_name,
                fm.framework_category,
                COUNT(*) as usage_count
            FROM framework_mappings fm
            JOIN impact_metrics im ON fm.impact_metric_id = im.id
            GROUP BY fm.framework_name, fm.framework_category
            ORDER BY usage_count DESC
        """)
        mappings = [dict(row) for row in cursor.fetchall()]
        
        # Coverage by framework name
        cursor.execute("""
            SELECT 
                fm.framework_name,
                COUNT(DISTINCT fm.impact_metric_id) as mapped_metrics,
                (SELECT COUNT(*) FROM impact_metrics) as total_metrics
            FROM framework_mappings fm
            GROUP BY fm.framework_name
        """)
        coverage = [dict(row) for row in cursor.fetchall()]
        
        for c in coverage:
            c['coverage_percentage'] = (c['mapped_metrics'] / c['total_metrics'] * 100) if c['total_metrics'] > 0 else 0
        
        return {
            "framework_mappings": mappings,
            "coverage_by_framework": coverage
        }

def _get_data_quality_analysis(db_path: str, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Get data quality analysis."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Verification status distribution
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN verification_accuracy >= 0.9 THEN 'High Quality'
                    WHEN verification_accuracy >= 0.7 THEN 'Medium Quality'
                    WHEN verification_accuracy >= 0.5 THEN 'Low Quality'
                    ELSE 'Poor Quality'
                END as quality_tier,
                COUNT(*) as count
            FROM impact_metrics
            WHERE verification_accuracy IS NOT NULL
            GROUP BY quality_tier
        """)
        quality_distribution = [dict(row) for row in cursor.fetchall()]
        
        # Missing data analysis
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN verification_accuracy IS NULL THEN 1 ELSE 0 END) as missing_accuracy,
                SUM(CASE WHEN metric_value IS NULL THEN 1 ELSE 0 END) as missing_values,
                SUM(CASE WHEN metric_unit IS NULL THEN 1 ELSE 0 END) as missing_units,
                COUNT(*) as total
            FROM impact_metrics
        """)
        missing_data = dict(cursor.fetchone())
        
        return {
            "quality_distribution": quality_distribution,
            "missing_data_analysis": missing_data
        }

def create_app(host: str = "0.0.0.0", port: int = 8000) -> FastAPI:
    """Create and configure the FastAPI app."""
    return app

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ImpactOS AI Web API Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    logger.info(f"Starting ImpactOS AI Web API on {args.host}:{args.port}")
    
    uvicorn.run(
        "web_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    ) 