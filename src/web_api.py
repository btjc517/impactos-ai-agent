"""
Web API service for ImpactOS AI system integration.

This module provides a FastAPI web service that exposes the ImpactOS AI system
functionality through REST API endpoints for web portal integration.
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# FastAPI and web service imports
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Local imports
from main import ImpactOSCLI
from query import QuerySystem, query_data
from schema import DatabaseSchema
from frameworks import get_framework_report, apply_framework_mappings
from verify import verify_all_data, verify_metric
from config import get_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question about the data")
    show_accuracy: bool = Field(False, description="Include verification accuracy in response")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="Answer to the question with citations")
    accuracy_summary: Optional[str] = Field(None, description="Data accuracy summary if requested")
    timestamp: datetime = Field(default_factory=datetime.now)

class IngestionRequest(BaseModel):
    file_path: str = Field(..., description="Path to file to ingest")
    file_type: Optional[str] = Field(None, description="File type override (auto-detected if not provided)")
    verify_after_ingestion: bool = Field(False, description="Run verification after ingestion")

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

# Initialize FastAPI app
app = FastAPI(
    title="ImpactOS AI API",
    description="REST API for ImpactOS AI social value data analysis system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for web portal integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ImpactOS CLI instance
impactos_cli = None

def get_cli_instance():
    """Get or create ImpactOS CLI instance."""
    global impactos_cli
    if impactos_cli is None:
        impactos_cli = ImpactOSCLI()
    return impactos_cli

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting ImpactOS AI Web API service...")
    cli = get_cli_instance()
    logger.info("ImpactOS AI Web API service ready")

@app.get("/", response_model=Dict[str, str])
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

@app.post("/query", response_model=QueryResponse)
async def query_data_endpoint(request: QueryRequest):
    """
    Query the system with natural language question.
    
    This is the main endpoint your web portal query bar should use.
    """
    try:
        cli = get_cli_instance()
        
        logger.info(f"Processing web API query: {request.question}")
        
        # Process the query
        answer = cli.query_data(request.question)
        
        # Get accuracy summary if requested
        accuracy_summary = None
        if request.show_accuracy:
            accuracy_summary = cli.get_verification_summary()
        
        return QueryResponse(
            answer=answer,
            accuracy_summary=accuracy_summary
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
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
        success = cli.ingest_data(request.file_path, request.file_type)
        
        # Run verification if requested
        verification_results = None
        if request.verify_after_ingestion and success:
            verification_results = verify_all_data(cli.db_path)
        
        return IngestionResponse(
            success=success,
            message="File ingested successfully" if success else "Ingestion failed",
            verification_results=verification_results
        )
        
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