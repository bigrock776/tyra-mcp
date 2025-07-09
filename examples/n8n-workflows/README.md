# 🔌 n8n Workflow Examples

This directory contains ready-to-import n8n workflow examples that demonstrate integration with the Tyra MCP Memory Server.

> **🕷️ For detailed Crawl4AI setup instructions, see [CRAWL4AI_SETUP.md](CRAWL4AI_SETUP.md)**

## 📂 Available Workflows

### 1. Crawl4AI Web Scraper → Tyra Memory
**File**: `web-scraper-memory-store.json`

**Purpose**: Automatically scrape web content using Crawl4AI semantic extraction and store it in Tyra Memory with advanced metadata and entity recognition

**Features**:
- **Crawl4AI Integration**: Semantic content extraction with AI-powered filtering
- **Enhanced Metadata**: Captures title, author, publish date, language, keywords
- **Entity Recognition**: Automatically extracts URLs, emails, dates, technical terms
- **Quality Filtering**: Only stores substantial, high-quality content
- **Confidence Scoring**: Provides reliability metrics for stored memories
- **Analytics Integration**: Tracks extraction success and content quality
- Content length filtering
- Automatic memory storage with metadata
- Error handling and logging

**Use Cases**:
- News aggregation
- Blog monitoring
- Documentation updates
- Competitive intelligence

**Configuration Required**:
```env
MEMORY_SERVER_URL=http://localhost:8000
TYRA_API_KEY=your-api-key
CRAWL4AI_URL=http://localhost:8001
```

### 2. Advanced Multi-Source Crawler
**File**: `crawl4ai-web-scraper-memory.json`

**Purpose**: Enterprise-grade web crawling with multiple extraction strategies, hallucination detection, and comprehensive analytics

**Features**:
- **Multi-URL Configuration**: Different strategies per site (HN, Reddit, ArXiv)
- **Advanced Entity Extraction**: Deep content analysis and metrics
- **Hallucination Detection**: Validates content grounding and confidence
- **Comprehensive Analytics**: Detailed logging and performance tracking
- **Provider Integration**: Full integration with Tyra's advanced RAG pipeline
- **Quality Assurance**: Multiple filtering stages and validation checks

**Use Cases**:
- Research paper monitoring
- Tech news aggregation
- Multi-source content analysis
- Academic literature tracking

**Configuration Required**:
```env
MEMORY_SERVER_URL=http://localhost:8000
TYRA_API_KEY=your-api-key
CRAWL4AI_URL=http://localhost:8001
```

### 3. Email Context Search & Response
**File**: `email-context-search.json`

**Purpose**: Process incoming emails and provide contextual responses

**Features**:
- Email webhook processing
- Context search in memory database
- Intelligent response generation
- Automatic email reply
- Memory storage for future reference

**Use Cases**:
- Customer support automation
- FAQ responses
- Email triage
- Knowledge base queries

**Configuration Required**:
```env
MEMORY_SERVER_URL=http://localhost:8000
EMAIL_FROM=your-email@domain.com
EMAIL_FALLBACK_CC=support@domain.com
```

### 4. Document Batch Processor
**File**: `document-batch-processor.json`

**Purpose**: Batch process documents and ingest them into memory

**Features**:
- Scheduled document processing
- Multiple file format support (TXT, PDF, DOCX, MD)
- Batch creation and processing
- Async processing for large datasets
- Comprehensive progress tracking

**Use Cases**:
- Document management
- Knowledge base building
- Archive processing
- Content migration

**Configuration Required**:
```env
MEMORY_SERVER_URL=http://localhost:8000
DOCUMENT_FOLDER_PATH=/path/to/documents
```

### 5. Customer Support Context Finder
**File**: `customer-support-context.json`

**Purpose**: Provide intelligent context for customer support tickets

**Features**:
- Support ticket webhook processing
- Knowledge base search
- Customer history lookup
- Intelligent routing (self-service vs escalation)
- Context delivery to support agents

**Use Cases**:
- Support ticket automation
- Agent assistance
- Customer history tracking
- Knowledge base utilization

**Configuration Required**:
```env
MEMORY_SERVER_URL=http://localhost:8000
SUPPORT_SYSTEM_URL=http://your-support-system.com
SUPPORT_API_TOKEN=your-support-api-token
```

## 🚀 Quick Start

### 1. Import Workflows
1. Open n8n interface
2. Click "Import from file"
3. Select desired workflow JSON file
4. Configure environment variables
5. Test and activate

### 2. Environment Setup
Create these environment variables in n8n:

```env
# Required for all workflows
MEMORY_SERVER_URL=http://localhost:8000

# For email workflows
EMAIL_FROM=noreply@yourdomain.com
EMAIL_FALLBACK_CC=support@yourdomain.com

# For document processing
DOCUMENT_FOLDER_PATH=/home/user/documents

# For support integration
SUPPORT_SYSTEM_URL=http://your-support-system.com
SUPPORT_API_TOKEN=your-api-token
```

### 3. Webhook URLs
After importing, note the webhook URLs for external integration:

- Email processor: `http://your-n8n.com/webhook/email-processor`
- Support tickets: `http://your-n8n.com/webhook/support-ticket`

## 🔧 Customization Guide

### Modifying Search Parameters
Adjust search behavior in memory search nodes:

```json
{
  "top_k": 10,           // Number of results
  "min_confidence": 0.6, // Minimum confidence threshold
  "rerank": true,        // Enable reranking
  "include_analysis": true // Include hallucination analysis
}
```

### Adding Metadata Filters
Filter searches by metadata:

```json
{
  "metadata_filters": {
    "categories": ["support", "documentation"],
    "customer_id": "customer_123",
    "exclude_sources": ["email"]
  }
}
```

### Adjusting Batch Sizes
Optimize performance by adjusting batch sizes:

```javascript
const batchSize = 20; // Increase for better performance
const maxFileSize = 5 * 1024 * 1024; // 5MB limit
```

### Custom Content Processing
Add custom content extraction logic:

```javascript
// Custom content cleaning
const cleanContent = content
  .replace(/\s+/g, ' ')           // Normalize whitespace
  .replace(/[^\w\s]/gi, '')       // Remove special chars
  .substring(0, 5000);            // Limit length
```

## 📊 Monitoring & Debugging

### Workflow Execution Logs
Check execution logs for:
- API response times
- Memory storage success rates
- Error patterns
- Performance metrics

### Common Issues

#### 1. Connection Timeouts
**Problem**: Workflows timing out on memory server requests
**Solution**: Increase timeout values in HTTP request nodes

```json
{
  "options": {
    "timeout": 60000  // 60 seconds
  }
}
```

#### 2. Rate Limiting
**Problem**: Too many requests to memory server
**Solution**: Add delays between requests

```javascript
// Add delay in code nodes
await new Promise(resolve => setTimeout(resolve, 1000)); // 1 second delay
```

#### 3. Memory Storage Failures
**Problem**: Memories not being stored properly
**Solution**: Check content length and format

```javascript
// Validate content before storage
if (content.length < 10) {
  throw new Error('Content too short');
}
```

### Performance Optimization

#### 1. Parallel Processing
Process multiple items in parallel:

```javascript
// Process batches concurrently
const promises = batches.map(batch => processBatch(batch));
const results = await Promise.all(promises);
```

#### 2. Caching Strategy
Implement caching to reduce API calls:

```javascript
// Simple in-memory cache
const cache = new Map();
const cacheKey = `search_${queryHash}`;

if (cache.has(cacheKey)) {
  return cache.get(cacheKey);
}
```

#### 3. Error Recovery
Implement retry logic:

```javascript
async function retryRequest(requestFn, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await requestFn();
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
    }
  }
}
```

## 🔗 Integration Examples

### Slack Integration
Extend workflows to integrate with Slack:

```json
{
  "name": "Slack Notification",
  "type": "n8n-nodes-base.slack",
  "parameters": {
    "channel": "#support",
    "text": "New high-priority ticket: {{ $json.ticket_id }}"
  }
}
```

### Database Integration
Store results in your database:

```json
{
  "name": "Store in Database",
  "type": "n8n-nodes-base.postgres",
  "parameters": {
    "query": "INSERT INTO workflow_results (ticket_id, confidence, strategy) VALUES ($1, $2, $3)",
    "values": ["{{ $json.ticket_id }}", "{{ $json.confidence }}", "{{ $json.strategy }}"]
  }
}
```

### Webhook Notifications
Notify external systems:

```json
{
  "name": "Notify External System",
  "type": "n8n-nodes-base.httpRequest",
  "parameters": {
    "method": "POST",
    "url": "https://your-system.com/webhook",
    "body": {
      "event": "memory_processed",
      "data": "{{ $json }}"
    }
  }
}
```

## 📚 Additional Resources

- [Tyra Memory Server API Documentation](../../API.md)
- [n8n Integration Guide](../../docs/N8N_INTEGRATION.md)
- [Installation Guide](../../INSTALLATION.md)
- [Configuration Reference](../../CONFIGURATION.md)

## 🆘 Support

If you encounter issues:

1. Check the workflow execution logs
2. Verify environment variables
3. Test individual nodes
4. Review the integration documentation
5. Check memory server health endpoint: `/v1/webhooks/health`

For additional help, refer to the main project documentation or create an issue in the project repository.