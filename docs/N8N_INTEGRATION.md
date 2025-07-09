# 🔌 n8n Integration Guide

This guide covers how to integrate n8n workflows with the Tyra MCP Memory Server for automated knowledge management and processing.

## 📋 Overview

The Tyra Memory Server provides comprehensive webhook endpoints that allow n8n workflows to:
- Store processed data as memories
- Search for relevant context
- Ingest documents automatically
- Process batches of information
- Handle real-time events

## 🛠️ Available Endpoints

### Memory Operations

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/webhooks/n8n/memory-store` | POST | Store workflow results as memories |
| `/v1/webhooks/n8n/memory-search` | POST | Search memories for context |
| `/v1/webhooks/ingest/document` | POST | Ingest documents into memory |
| `/v1/webhooks/ingest/batch` | POST | Batch process multiple memories |
| `/v1/webhooks/events/memory-update` | POST | Handle memory update events |
| `/v1/webhooks/notify` | POST | Send notifications to external systems |

## 🔧 Integration Patterns

### Pattern 1: Web Scraping → Memory Storage

**Use Case**: Automatically store scraped content as memories

**Workflow**:
1. **HTTP Request** → Scrape website content
2. **Code** → Clean and structure data
3. **HTTP Request** → POST to `/v1/webhooks/n8n/memory-store`

**Configuration**:
```json
{
  "workflow_id": "web-scraper-001",
  "execution_id": "{{ $json.execution_id }}",
  "node_name": "Memory Store",
  "data": {
    "content": "{{ $json.scraped_content }}",
    "metadata": {
      "url": "{{ $json.source_url }}",
      "scraped_at": "{{ $now }}",
      "content_type": "web_article"
    }
  },
  "agent_id": "web-scraper",
  "session_id": "scraping-session-{{ $now.format('YYYY-MM-DD') }}"
}
```

### Pattern 2: Email Processing → Context Retrieval

**Use Case**: Process incoming emails and find relevant context

**Workflow**:
1. **Email Trigger** → New email received
2. **HTTP Request** → POST to `/v1/webhooks/n8n/memory-search`
3. **Code** → Format search results
4. **Email** → Send response with context

**Configuration**:
```json
{
  "workflow_id": "email-processor-001",
  "execution_id": "{{ $json.execution_id }}",
  "node_name": "Context Search",
  "data": {
    "query": "{{ $json.email_subject }} {{ $json.email_body }}",
    "top_k": 5,
    "min_confidence": 0.7,
    "rerank": true,
    "include_analysis": true
  },
  "agent_id": "email-assistant",
  "session_id": "email-{{ $json.email_id }}"
}
```

### Pattern 3: Document Processing Pipeline

**Use Case**: Automated document ingestion and processing

**Workflow**:
1. **File Trigger** → New document uploaded
2. **Code** → Extract text content
3. **HTTP Request** → POST to `/v1/webhooks/ingest/document`
4. **HTTP Request** → POST to `/v1/webhooks/notify` (optional)

**Configuration**:
```json
{
  "content": "{{ $json.document_text }}",
  "title": "{{ $json.filename }}",
  "source": "document-processor",
  "agent_id": "doc-processor",
  "metadata": {
    "filename": "{{ $json.filename }}",
    "file_size": "{{ $json.file_size }}",
    "upload_date": "{{ $now }}",
    "processor_version": "1.0"
  },
  "chunk_content": true,
  "extract_entities": true
}
```

### Pattern 4: Batch Data Processing

**Use Case**: Process multiple data items efficiently

**Workflow**:
1. **Schedule Trigger** → Daily/hourly processing
2. **Database** → Fetch unprocessed items
3. **Code** → Format batch request
4. **HTTP Request** → POST to `/v1/webhooks/ingest/batch`

**Configuration**:
```json
{
  "memories": [
    {
      "content": "{{ item.text }}",
      "metadata": {
        "source_id": "{{ item.id }}",
        "category": "{{ item.category }}"
      },
      "extract_entities": true,
      "chunk_content": false
    }
  ],
  "agent_id": "batch-processor",
  "session_id": "batch-{{ $now.format('YYYY-MM-DD-HH') }}",
  "process_async": true
}
```

### Pattern 5: Event-Driven Memory Updates

**Use Case**: React to external system changes

**Workflow**:
1. **Webhook Trigger** → External system event
2. **Code** → Transform event data
3. **HTTP Request** → POST to `/v1/webhooks/events/memory-update`

**Configuration**:
```json
{
  "event_type": "memory_updated",
  "source": "external-system",
  "timestamp": "{{ $now }}",
  "data": {
    "memory_id": "{{ $json.memory_id }}",
    "changes": "{{ $json.changes }}",
    "updated_by": "{{ $json.user_id }}"
  },
  "metadata": {
    "system": "external-system",
    "event_id": "{{ $json.event_id }}"
  }
}
```

## 🔐 Authentication & Security

### Authentication Headers
```http
Content-Type: application/json
X-Agent-ID: your-agent-id
X-Session-ID: your-session-id (optional)
Authorization: Bearer your-token (if enabled)
```

### Security Best Practices

1. **Agent Isolation**: Use unique `agent_id` for each workflow type
2. **Session Management**: Use meaningful `session_id` values for tracking
3. **Input Validation**: Always validate webhook data before processing
4. **Rate Limiting**: Be aware of rate limits on webhook endpoints
5. **Error Handling**: Implement proper error handling and retries

## 📊 Monitoring & Debugging

### Response Format
All webhook endpoints return a standardized response:

```json
{
  "success": true,
  "message": "Operation completed successfully",
  "event_id": "uuid-v4",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "memory_id": "mem_123",
    "entities_created": 5,
    "processing_time": {
      "total": 0.245,
      "embedding": 0.123,
      "storage": 0.089
    }
  }
}
```

### Health Check
Monitor webhook health:
```http
GET /v1/webhooks/health
```

### Common Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| 400 | Bad Request - Invalid data format | Check request schema |
| 401 | Unauthorized - Invalid credentials | Verify authentication |
| 429 | Rate Limited - Too many requests | Implement backoff |
| 500 | Server Error - Internal failure | Check server logs |

## 🚀 Performance Optimization

### Batch Processing
- Use `/ingest/batch` for multiple items
- Set `process_async: true` for large batches
- Limit batch size to 100 items maximum

### Content Optimization
- Enable `chunk_content` for large documents (>1000 chars)
- Use `extract_entities: false` for simple text storage
- Include relevant metadata for better searchability

### Caching Strategy
- Use consistent `session_id` for related operations
- Memory searches are cached for 1 hour
- Embeddings are cached for 24 hours

## 🔄 Workflow Templates

### Template 1: Smart RSS Feed Processor
```json
{
  "name": "Smart RSS Processor",
  "nodes": [
    {
      "name": "RSS Trigger",
      "type": "n8n-nodes-base.rssFeedRead",
      "parameters": {
        "url": "{{ $env.RSS_FEED_URL }}"
      }
    },
    {
      "name": "Content Cleanup",
      "type": "n8n-nodes-base.code",
      "parameters": {
        "jsCode": "// Clean HTML and extract key content\nconst content = $input.item.json.content;\nconst cleanText = content.replace(/<[^>]*>/g, '');\nreturn { clean_content: cleanText, original: $input.item.json };"
      }
    },
    {
      "name": "Store Memory",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "POST",
        "url": "{{ $env.MEMORY_SERVER_URL }}/v1/webhooks/n8n/memory-store",
        "jsonParameters": true,
        "bodyParameters": {
          "workflow_id": "rss-processor",
          "execution_id": "{{ $execution.id }}",
          "node_name": "Store Memory",
          "data": {
            "content": "{{ $json.clean_content }}",
            "metadata": {
              "title": "{{ $node[\"RSS Trigger\"].json.title }}",
              "url": "{{ $node[\"RSS Trigger\"].json.link }}",
              "published": "{{ $node[\"RSS Trigger\"].json.pubDate }}"
            }
          },
          "agent_id": "rss-processor"
        }
      }
    }
  ]
}
```

### Template 2: Customer Support Context Finder
```json
{
  "name": "Support Context Finder",
  "nodes": [
    {
      "name": "Support Ticket Trigger",
      "type": "n8n-nodes-base.webhook",
      "parameters": {
        "path": "support-ticket"
      }
    },
    {
      "name": "Search Relevant Context",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "POST",
        "url": "{{ $env.MEMORY_SERVER_URL }}/v1/webhooks/n8n/memory-search",
        "jsonParameters": true,
        "bodyParameters": {
          "workflow_id": "support-context",
          "execution_id": "{{ $execution.id }}",
          "node_name": "Context Search",
          "data": {
            "query": "{{ $json.ticket_description }} {{ $json.customer_id }}",
            "top_k": 10,
            "min_confidence": 0.6,
            "rerank": true
          },
          "agent_id": "support-assistant"
        }
      }
    },
    {
      "name": "Format Response",
      "type": "n8n-nodes-base.code",
      "parameters": {
        "jsCode": "// Format search results for support agent\nconst results = $json.data.results;\nconst formatted = results.map(r => `${r.content} (Confidence: ${r.confidence})`);\nreturn { context: formatted.join('\\n\\n'), ticket_id: $node[\"Support Ticket Trigger\"].json.ticket_id };"
      }
    }
  ]
}
```

## 🔗 Advanced Integration

### Webhook Chaining
Chain multiple webhook calls for complex workflows:

1. **Store** → Store initial data
2. **Search** → Find related context  
3. **Store** → Store enriched data with context
4. **Notify** → Inform external systems

### Error Recovery
Implement robust error handling:

```javascript
// n8n Code node for error handling
try {
  const response = await $http.request({
    method: 'POST',
    url: 'memory-server/webhook/endpoint',
    data: payload
  });
  
  if (!response.success) {
    throw new Error(response.message);
  }
  
  return response;
} catch (error) {
  // Log error and implement fallback
  console.error('Webhook failed:', error);
  
  // Store in error queue for retry
  await $http.request({
    method: 'POST',
    url: 'error-queue/endpoint',
    data: { original_payload: payload, error: error.message }
  });
  
  return { success: false, error: error.message };
}
```

### Performance Monitoring
Track webhook performance in n8n:

```javascript
// Performance tracking code
const startTime = Date.now();

const response = await $http.request({
  method: 'POST',
  url: 'memory-server-endpoint',
  data: payload
});

const duration = Date.now() - startTime;

// Log performance metrics
console.log(`Webhook completed in ${duration}ms`);

if (duration > 5000) {
  console.warn('Slow webhook detected');
}

return { ...response, processing_time: duration };
```

## 📚 Additional Resources

- [Tyra Memory Server API Documentation](../API.md)
- [n8n Webhook Documentation](https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.webhook/)
- [Installation Guide](../INSTALLATION.md)
- [Configuration Reference](../CONFIGURATION.md)