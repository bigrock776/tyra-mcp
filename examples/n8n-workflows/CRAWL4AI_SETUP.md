# 🕷️ Crawl4AI n8n Workflow Setup Guide

## 📋 Overview

This guide shows you how to set up and use the enhanced Crawl4AI-powered n8n workflows to automatically extract web content and store it in Tyra Memory MCP.

## 🎯 What You Get

### Advanced Web Crawling
- **Semantic Content Extraction** - Uses AI to identify and extract relevant content
- **Intelligent Chunking** - Automatically breaks content into meaningful segments
- **Entity Recognition** - Extracts URLs, emails, dates, technical terms
- **Quality Filtering** - Only stores high-quality, substantial content
- **Metadata Enrichment** - Captures author, publish date, language, keywords

### Tyra Memory Integration
- **Enhanced Storage** - Stores content with rich metadata and embeddings
- **Entity Extraction** - Automatically extracts entities and relationships
- **Hallucination Detection** - Validates content quality and grounding
- **Confidence Scoring** - Provides confidence levels for stored memories
- **Analytics Integration** - Tracks performance and optimization metrics

## 🛠️ Prerequisites

### 1. Crawl4AI Server
You need a running Crawl4AI server. Install and start it:

```bash
# Install crawl4ai
pip install crawl4ai

# Start the server (default port 8001)
crawl4ai-server --host 0.0.0.0 --port 8001
```

### 2. Tyra Memory Server
Ensure your Tyra MCP Memory Server is running:

```bash
# Start Tyra Memory Server
cd tyra-mcp-memory-server
source venv/bin/activate
python main.py

# OR start API server
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### 3. Environment Variables

Set these environment variables in n8n:

```bash
# Tyra Memory Server
MEMORY_SERVER_URL=http://localhost:8000
TYRA_API_KEY=your-api-key-here

# Crawl4AI Server  
CRAWL4AI_URL=http://localhost:8001
```

## 📥 Available Workflows

### 1. Basic Crawl4AI Web Scraper (`web-scraper-memory-store.json`)

**Features:**
- Scheduled crawling of predefined URLs
- Crawl4AI semantic extraction
- Quality filtering and content validation
- Direct storage to Tyra Memory with metadata

**Target Sites:**
- Hacker News (tech news)
- Reddit Machine Learning (research discussions)

### 2. Advanced Multi-Source Crawler (`crawl4ai-web-scraper-memory.json`)

**Features:**
- Multiple URL configurations with different strategies
- Entity extraction and content metrics
- Hallucination detection and confidence analysis
- Comprehensive analytics and logging

**Target Sites:**
- Hacker News
- Reddit Machine Learning
- ArXiv AI Papers

## 🚀 Installation Steps

### 1. Import Workflows

In n8n:
1. Go to **Workflows** → **Import from File**
2. Select one of the workflow JSON files
3. Click **Import**

### 2. Configure Environment Variables

In n8n Settings → Environment Variables:
```
MEMORY_SERVER_URL = http://localhost:8000
TYRA_API_KEY = your-generated-api-key
CRAWL4AI_URL = http://localhost:8001
```

### 3. Update API Credentials

In the workflow:
1. Click on "Store in Tyra Memory" node
2. Update the **X-API-Key** header with your actual API key
3. Verify the **MEMORY_SERVER_URL** points to your server

### 4. Customize URLs and Extraction

Edit the URL configuration in the workflow:

```javascript
const urlsToScrape = [
  {
    url: 'https://your-target-site.com',
    category: 'your_category',
    max_depth: 1,
    follow_links: true,
    css_selector: '.your-content-selector'
  }
  // Add more URLs...
];
```

## ⚙️ Configuration Options

### Crawl4AI Extraction Strategy

```json
{
  "extraction_strategy": {
    "type": "CosineStrategy",
    "semantic_filter": "tech news, AI, machine learning, research",
    "word_count_threshold": 100,
    "max_dist": 0.2,
    "linkage_method": "ward",
    "top_k": 3
  }
}
```

**Options:**
- `type`: `CosineStrategy`, `LLMExtractionStrategy`, `JsonCssExtractionStrategy`
- `semantic_filter`: Keywords for content relevance
- `word_count_threshold`: Minimum words per chunk
- `max_dist`: Maximum semantic distance for clustering
- `top_k`: Number of top clusters to return

### Content Chunking

```json
{
  "chunking_strategy": {
    "type": "RegexChunking",
    "patterns": ["\\n\\n", "\\. "],
    "min_length": 100,
    "max_length": 1000
  }
}
```

### Quality Filters

The workflow includes multiple quality gates:

1. **Content Length**: Minimum 200 characters
2. **Word Count**: Minimum 30 words  
3. **Error Handling**: Skips failed extractions
4. **Duplicate Detection**: Prevents storing identical content

## 📊 Monitoring and Analytics

### Success Metrics

The workflow tracks:
- **Memory Storage Success Rate**
- **Content Quality Scores** 
- **Semantic Similarity Scores**
- **Entity Extraction Counts**
- **Processing Times**
- **Hallucination Detection Results**

### Logging

View logs in:
- n8n execution logs
- Tyra Memory Server logs (`logs/memory-server.log`)
- Crawl4AI server logs

### Analytics Dashboard

Access workflow analytics:
```bash
curl http://localhost:8000/v1/admin/analytics/workflow-stats \
  -H "X-API-Key: your-api-key"
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Crawl4AI Connection Failed
```
Error: ECONNREFUSED localhost:8001
```

**Solution:**
- Ensure Crawl4AI server is running
- Check if port 8001 is available
- Update `CRAWL4AI_URL` if using different port

#### 2. Tyra Memory API Errors
```
Error: 401 Unauthorized
```

**Solution:**
- Verify `TYRA_API_KEY` is correct
- Check API key has proper permissions
- Ensure Tyra Memory Server is running

#### 3. No Content Extracted
```
Warning: Content filtered out (too short)
```

**Solution:**
- Adjust `word_count_threshold` in extraction strategy
- Update CSS selectors for target sites
- Check if target sites have anti-bot measures

#### 4. High Memory Usage

**Solution:**
- Reduce `max_depth` in crawl configuration
- Limit number of URLs per execution
- Increase chunking thresholds

### Debug Mode

Enable debug logging:

1. In n8n workflow, add debug nodes
2. Set Tyra Memory Server to DEBUG level:
   ```bash
   export LOG_LEVEL=DEBUG
   python main.py
   ```

## 🎛️ Advanced Configuration

### Custom Content Selectors

For specific websites, customize CSS selectors:

```javascript
const siteConfigs = {
  'news.ycombinator.com': {
    content_selector: '.titleline a, .subtext',
    exclude_selectors: ['.reply', '.navs']
  },
  'reddit.com': {
    content_selector: '.Post, .Comment',
    exclude_selectors: ['.promotedlink']
  }
};
```

### Webhook Integration

Set up webhook triggers for real-time crawling:

```bash
# Trigger workflow via webhook
curl -X POST http://your-n8n-server/webhook/crawl4ai-trigger \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://example.com"], "immediate": true}'
```

### Batch Processing

For large-scale crawling:

1. Split URLs into batches
2. Use parallel execution
3. Implement rate limiting
4. Add retry mechanisms

## 📈 Performance Optimization

### Best Practices

1. **Rate Limiting**: Add delays between requests
2. **Caching**: Enable Crawl4AI caching
3. **Filtering**: Use strict quality filters
4. **Chunking**: Optimize chunk sizes for your content
5. **Monitoring**: Track success rates and adjust parameters

### Resource Management

```javascript
// Optimized configuration for high-volume crawling
const optimizedConfig = {
  delay_between_requests: 2000,
  max_pages_per_domain: 50,
  timeout: 30000,
  cache_mode: "enabled",
  respect_robots_txt: true
};
```

## 📚 Additional Resources

- [Crawl4AI Documentation](https://github.com/unclecode/crawl4ai)
- [n8n Workflow Documentation](https://docs.n8n.io/workflows/)
- [Tyra Memory API Reference](../../API.md)
- [Configuration Guide](../../CONFIGURATION.md)

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review n8n execution logs
3. Monitor Tyra Memory Server health: `curl http://localhost:8000/health`
4. Verify Crawl4AI server status: `curl http://localhost:8001/health`

---

🎉 **Happy Crawling!** Your automated knowledge extraction pipeline is now ready to continuously feed high-quality content into Tyra Memory.