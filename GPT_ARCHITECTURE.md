# ImpactOS GPT Tools & Multi-Agent Architecture

## Overview

This document describes the new architecture that combines OpenAI's GPT tools with a Claude-based multi-agent system for enhanced data processing and querying capabilities.

## Architecture Components

### 1. GPT Tools Module (`src/gpt_tools/`)

Leverages OpenAI's native capabilities:

- **AssistantManager**: Manages GPT-4 assistants with file search and code interpreter
- **FileManager**: Handles file uploads and storage for OpenAI API
- **EmbeddingService**: Generates embeddings using text-embedding-3 models
- **RetrievalService**: Hybrid retrieval combining semantic and keyword search

### 2. Multi-Agent System (`src/agents/`)

Three specialized agents working in coordination:

#### Architect Agent (Claude Opus 4.1)
- **Role**: System architect and orchestrator
- **Responsibilities**:
  - Analyze user requests and create execution plans
  - Delegate tasks to specialist agents
  - Review results for quality and completeness
  - Make strategic decisions

#### Data Agent (Claude Sonnet 3.5)
- **Role**: Data processing specialist
- **Responsibilities**:
  - Ingest files (CSV, Excel, PDF)
  - Extract structured metrics
  - Validate data quality
  - Map to frameworks (SDGs, MAC, TOMs, B Corp)

#### Query Agent (Claude Sonnet 3.5)
- **Role**: Query processing specialist
- **Responsibilities**:
  - Understand natural language questions
  - Search and retrieve information
  - Generate accurate answers with citations
  - Recommend visualizations

### 3. Communication Protocol

Agents communicate through:
- **Shared Context** (`agents/context.md`): Markdown file for shared information
- **Task Queue** (`agents/tasks.json`): JSON-based task management
- **Message System** (`agents/messages.json`): Inter-agent messaging
- **System State** (`agents/system_state.json`): Global state tracking

## Usage

### Prerequisites

```bash
# Required API Keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Install dependencies
pip install -r requirements.txt
```

### Command Line Interface

The new CLI (`src/main_gpt.py`) provides enhanced commands:

```bash
# Ingest data with multi-agent system
python src/main_gpt.py ingest data/sample.xlsx

# Query with multi-agent system
python src/main_gpt.py query "What are our carbon emissions?"

# Batch ingest entire directory
python src/main_gpt.py batch-ingest data/

# Interactive mode
python src/main_gpt.py interactive

# System status
python src/main_gpt.py status

# Use direct GPT tools (bypass agents)
python src/main_gpt.py ingest data/sample.xlsx --no-agents
python src/main_gpt.py query "Your question" --no-agents
```

### Testing

Run the integration tests:

```bash
python test_gpt_integration.py
```

## How It Works

### Data Ingestion Flow

1. **User Request**: User provides file for ingestion
2. **Architect Analysis**: Architect agent analyzes requirements
3. **Task Creation**: Architect creates tasks for Data Agent
4. **GPT Processing**: Data Agent uses GPT-4 for extraction
5. **Validation**: Data Agent validates extracted metrics
6. **Framework Mapping**: Metrics mapped to sustainability frameworks
7. **Review**: Architect reviews results for quality

### Query Processing Flow

1. **User Question**: Natural language question received
2. **Intent Analysis**: Query Agent analyzes question intent
3. **GPT Search**: Uses GPT file search and retrieval
4. **Answer Generation**: Combines GPT and Claude capabilities
5. **Citation Addition**: Adds proper source citations
6. **Visualization**: Recommends charts if helpful
7. **Quality Review**: Architect validates answer

## Benefits

### Over Pure Custom Implementation
- **Reduced Maintenance**: Leverage OpenAI's maintained infrastructure
- **Better Embeddings**: OpenAI's embeddings outperform open-source models
- **File Search**: Native file search with citations
- **Code Interpreter**: Built-in data analysis capabilities

### Over Pure GPT Implementation
- **Multi-Agent Specialization**: Different agents for different tasks
- **Quality Control**: Architect reviews all outputs
- **Framework Expertise**: Specialized knowledge for impact frameworks
- **Flexible Architecture**: Can adjust agent roles and models

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Optional
IMPACTOS_DB_PATH=db/impactos.db  # Database location
```

### Agent Models

- **Architect**: Claude 3 Opus (highest capability)
- **Data/Query Agents**: Claude 3.5 Sonnet (balance of speed and capability)

Models can be changed in agent initialization:
```python
# In architect_agent.py
model="claude-3-opus-20240229"  # Can update to newer versions
```

## Deployment on Render

The system is designed to deploy on Render with minimal changes:

1. Set environment variables in Render dashboard
2. Deploy using existing `render.yaml`
3. The web API automatically uses the new architecture

## Monitoring

### System Status

Check system health:
```python
orchestrator.get_system_status()
```

Returns:
- Active agents and their status
- Task queue statistics
- Recent activity
- Performance metrics

### Agent Communication

Monitor inter-agent communication:
- Check `agents/context.md` for shared context
- Review `agents/tasks.json` for task queue
- Examine `agents/messages.json` for messages

## Troubleshooting

### Common Issues

1. **Missing API Keys**
   - Ensure both OPENAI_API_KEY and ANTHROPIC_API_KEY are set
   - System can run with just one, but with limited functionality

2. **Agent Communication Failures**
   - Check `agents/` directory permissions
   - Ensure JSON files are valid
   - Review agent logs for errors

3. **GPT Rate Limits**
   - Implement exponential backoff
   - Use caching for repeated queries
   - Consider upgrading OpenAI tier

## Future Enhancements

- [ ] Add more specialized agents (e.g., Visualization Agent)
- [ ] Implement agent learning from feedback
- [ ] Add real-time collaboration features
- [ ] Integrate with more data sources
- [ ] Enhanced visualization generation
- [ ] Multi-language support

## Contributing

When adding new features:
1. Consider which agent should handle the task
2. Update communication protocol if needed
3. Add appropriate tests
4. Document agent interactions

## License

[Your License Here]