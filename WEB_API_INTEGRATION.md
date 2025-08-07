# ImpactOS AI Web API Integration Guide

This guide explains how to connect your web portal to the ImpactOS AI system using the REST API service.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Web API Service

```bash
# Development mode (with auto-reload)
./start_web_api.sh

# Production mode
./start_web_api.sh --mode production --port 8080

# Custom configuration
./start_web_api.sh --host localhost --port 8001 --log-level debug
```

### 3. Test the Service

```bash
# Check health
curl http://localhost:8000/health

# Test query endpoint
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "How much was donated to charity?"}'
```

## API Endpoints

### Core Query Endpoint (Main Portal Integration)

**POST `/query`** - Process natural language questions

```javascript
// Example web portal integration
const queryData = async (question) => {
    const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            question: question,
            show_accuracy: true  // Optional: include data accuracy
        })
    });
    
    const result = await response.json();
    return result;
};

// Usage in your web portal
const answer = await queryData("What is our total carbon footprint?");
console.log(answer.answer);  // AI-generated answer with citations
console.log(answer.accuracy_summary);  // Data accuracy info
```

### Health and Status

**GET `/health`** - Service health check

```javascript
const checkHealth = async () => {
    const response = await fetch('http://localhost:8000/health');
    const health = await response.json();
    return health;
};
```

### Data Management

**GET `/data`** - List available and ingested data

```javascript
const getDataSources = async () => {
    const response = await fetch('http://localhost:8000/data');
    const data = await response.json();
    return data;
};
```

**POST `/upload`** - Upload and process files

```javascript
const uploadFile = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('http://localhost:8000/upload?verify_after_ingestion=true', {
        method: 'POST',
        body: formData
    });
    
    return await response.json();
};
```

### Framework Reports

**GET `/frameworks`** - Get framework mapping reports

```javascript
const getFrameworkReport = async () => {
    const response = await fetch('http://localhost:8000/frameworks');
    const report = await response.json();
    return report;
};
```

## Frontend Integration Examples

### React/Next.js Integration

```jsx
import React, { useState } from 'react';

const QueryComponent = () => {
    const [question, setQuestion] = useState('');
    const [answer, setAnswer] = useState('');
    const [loading, setLoading] = useState(false);

    const handleQuery = async () => {
        setLoading(true);
        try {
            const response = await fetch('http://localhost:8000/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: question,
                    show_accuracy: true
                })
            });
            
            const result = await response.json();
            setAnswer(result.answer);
        } catch (error) {
            console.error('Query failed:', error);
            setAnswer('Error processing query');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="query-component">
            <div className="query-bar">
                <input
                    type="text"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="Ask about your social value data..."
                    className="query-input"
                />
                <button onClick={handleQuery} disabled={loading}>
                    {loading ? 'Processing...' : 'Ask'}
                </button>
            </div>
            
            {answer && (
                <div className="answer-section">
                    <h3>Answer:</h3>
                    <pre>{answer}</pre>
                </div>
            )}
        </div>
    );
};

export default QueryComponent;
```

### Vue.js Integration

```vue
<template>
  <div class="impactos-query">
    <div class="query-bar">
      <input
        v-model="question"
        @keyup.enter="submitQuery"
        placeholder="Ask about your social value data..."
        :disabled="loading"
      />
      <button @click="submitQuery" :disabled="loading || !question">
        {{ loading ? 'Processing...' : 'Ask' }}
      </button>
    </div>
    
    <div v-if="answer" class="answer-section">
      <h3>Answer:</h3>
      <div class="answer-content" v-html="formatAnswer(answer.answer)"></div>
      <div v-if="answer.accuracy_summary" class="accuracy-info">
        <small>{{ answer.accuracy_summary }}</small>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ImpactOSQuery',
  data() {
    return {
      question: '',
      answer: null,
      loading: false,
      apiBaseUrl: 'http://localhost:8000'
    };
  },
  methods: {
    async submitQuery() {
      if (!this.question.trim()) return;
      
      this.loading = true;
      try {
        const response = await fetch(`${this.apiBaseUrl}/query`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            question: this.question,
            show_accuracy: true
          })
        });
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        this.answer = await response.json();
      } catch (error) {
        console.error('Query failed:', error);
        this.answer = {
          answer: 'Error processing your question. Please try again.',
          accuracy_summary: null
        };
      } finally {
        this.loading = false;
      }
    },
    
    formatAnswer(answer) {
      // Basic formatting for citations and line breaks
      return answer
        .replace(/\[(\d+)\]/g, '<sup>[$1]</sup>')
        .replace(/\n/g, '<br>');
    }
  }
};
</script>
```

### Plain JavaScript Integration

```html
<!DOCTYPE html>
<html>
<head>
    <title>ImpactOS AI Portal</title>
    <style>
        .query-container {
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            font-family: Arial, sans-serif;
        }
        
        .query-bar {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .query-input {
            flex: 1;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 6px;
            font-size: 16px;
        }
        
        .query-button {
            padding: 12px 24px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
        }
        
        .query-button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        
        .answer-section {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <div class="query-container">
        <h1>ImpactOS AI Query Portal</h1>
        
        <div class="query-bar">
            <input type="text" id="questionInput" class="query-input" 
                   placeholder="Ask about your social value data..." />
            <button id="queryButton" class="query-button">Ask</button>
        </div>
        
        <div id="answerSection" class="answer-section" style="display: none;"></div>
    </div>

    <script>
        const API_BASE_URL = 'http://localhost:8000';
        
        const questionInput = document.getElementById('questionInput');
        const queryButton = document.getElementById('queryButton');
        const answerSection = document.getElementById('answerSection');
        
        async function submitQuery() {
            const question = questionInput.value.trim();
            if (!question) return;
            
            queryButton.disabled = true;
            queryButton.textContent = 'Processing...';
            answerSection.style.display = 'none';
            
            try {
                const response = await fetch(`${API_BASE_URL}/query`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        question: question,
                        show_accuracy: true
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const result = await response.json();
                
                let answerText = result.answer;
                if (result.accuracy_summary) {
                    answerText += `\n\nData Accuracy: ${result.accuracy_summary}`;
                }
                
                answerSection.textContent = answerText;
                answerSection.style.display = 'block';
                
            } catch (error) {
                console.error('Query failed:', error);
                answerSection.textContent = 'Error processing your question. Please try again.';
                answerSection.style.display = 'block';
            } finally {
                queryButton.disabled = false;
                queryButton.textContent = 'Ask';
            }
        }
        
        queryButton.addEventListener('click', submitQuery);
        questionInput.addEventListener('keyup', function(event) {
            if (event.key === 'Enter') {
                submitQuery();
            }
        });
    </script>
</body>
</html>
```

## Configuration

### Environment Variables

```bash
# Required for GPT-4 functionality
export OPENAI_API_KEY="your-openai-api-key"

# Optional: Custom database path
export IMPACTOS_DB_PATH="path/to/your/database.db"
```

### CORS Configuration

For production, configure CORS in `src/web_api.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-portal-domain.com"],  # Specific domains
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### API Configuration

The system uses configuration from `config/system_config.json`. Key settings:

- `query_processing.max_results_for_gpt`: Maximum results for GPT-4 processing
- `query_processing.gpt4_model`: GPT-4 model to use
- `vector_search.min_similarity_threshold`: Minimum similarity for vector search

## Security Considerations

### Production Deployment

1. **Environment Variables**: Use environment variables for sensitive configuration
2. **CORS**: Configure specific allowed origins
3. **Rate Limiting**: Implement rate limiting for API endpoints
4. **Authentication**: Add authentication middleware if required
5. **HTTPS**: Use HTTPS in production

### Example Nginx Configuration

```nginx
server {
    listen 80;
    server_name your-api-domain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Monitoring and Logging

### Health Monitoring

```javascript
// Check API health regularly
const monitorHealth = async () => {
    try {
        const response = await fetch('http://localhost:8000/health');
        const health = await response.json();
        
        if (health.status !== 'healthy') {
            console.error('API health check failed:', health);
        }
        
        return health;
    } catch (error) {
        console.error('API unavailable:', error);
        return null;
    }
};

// Check every 30 seconds
setInterval(monitorHealth, 30000);
```

### Error Handling

```javascript
const handleApiError = (error, response) => {
    if (response.status === 500) {
        return 'Server error. Please try again later.';
    } else if (response.status === 404) {
        return 'API endpoint not found.';
    } else if (response.status >= 400) {
        return 'Request error. Please check your input.';
    } else {
        return 'Network error. Please check your connection.';
    }
};
```

## API Response Examples

### Query Response

```json
{
    "answer": "Based on the comprehensive data analysis, here's what I found:\n\nAGGREGATED TOTALS:\n1. Donations: Total Charitable Donations = 15000 GBP (from 3 records) | Frameworks: UK Social Value Model: A1, A2; UN SDGs: SDG 1, SDG 10\n\nSources with Details: TakingCare_Payroll_Synthetic_Data.xlsx (Sheet: Donations; Cell: D5-D7); TakingCare_Benevity_Synthetic_Data.xlsx (Sheet: Charity; Cell: B10-B15)",
    "accuracy_summary": "12/15 metrics verified (87.3% accuracy)",
    "timestamp": "2024-01-15T10:30:45.123456"
}
```

### Health Response

```json
{
    "status": "healthy",
    "version": "1.0.0",
    "database_connected": true,
    "openai_configured": true,
    "timestamp": "2024-01-15T10:30:45.123456"
}
```

### Data List Response

```json
{
    "available_files": [
        {
            "filename": "TakingCare_Payroll_Synthetic_Data.xlsx",
            "size_bytes": 45632,
            "modified": "2024-01-15T09:15:30"
        }
    ],
    "ingested_sources": [
        {
            "filename": "TakingCare_Payroll_Synthetic_Data.xlsx",
            "processed_timestamp": "2024-01-15T09:20:15",
            "processing_status": "completed",
            "metric_count": 25,
            "avg_accuracy": 0.89
        }
    ],
    "timestamp": "2024-01-15T10:30:45.123456"
}
```

## Troubleshooting

### Common Issues

1. **CORS Errors**: Configure allowed origins in web_api.py
2. **OpenAI API Errors**: Check OPENAI_API_KEY environment variable
3. **Database Not Found**: Ensure database is initialized with `python src/main.py schema`
4. **Import Errors**: Run the API from the project root directory

### Debugging

```bash
# Start with debug logging
./start_web_api.sh --log-level debug

# Check API documentation
# Visit http://localhost:8000/docs for interactive API docs
```

## Performance Optimization

### Caching

The system includes built-in caching for query results. Configure in `config/system_config.json`:

```json
{
    "query_processing": {
        "enable_result_caching": true,
        "cache_ttl_seconds": 3600
    }
}
```

### Scaling

For high-traffic deployments:

1. Use multiple worker processes
2. Implement load balancing
3. Consider caching layer (Redis)
4. Database connection pooling

```bash
# Production with multiple workers
./start_web_api.sh --mode production --port 8000
```

This will start the API with 4 worker processes optimized for production use. 