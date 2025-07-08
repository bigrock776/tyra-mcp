# 🤖 Tyra Integration Guide

## 📋 Overview

This guide explains how to integrate Tyra AI agent with the advanced memory server, including specialized trading configurations, risk management features, and high-confidence requirements.

## 🎯 Tyra-Specific Features

### Agent Configuration

Tyra has specialized configuration optimized for trading and financial analysis:

```yaml
# config/agents.yaml
tyra:
  display_name: "Tyra AI Agent"
  description: "Primary AI trading and analysis agent"

  memory_settings:
    max_memories: 100000         # Large memory capacity
    retention_days: 365          # Full year retention
    auto_cleanup: true           # Automatic cleanup enabled

  confidence_thresholds:
    trading_actions: 95          # Very high threshold for trades
    analysis_output: 80          # High threshold for analysis
    general_responses: 60        # Standard threshold for general queries

  tools:
    - sentiment_analysis
    - technical_indicators
    - news_analysis
    - risk_assessment

  preferences:
    response_style: "concise"
    include_sources: true
    explain_confidence: true
```

### High-Confidence Requirements

Tyra requires the highest confidence levels for trading operations:

- **Trading Actions**: 95% confidence minimum
- **Risk Assessment**: 90% confidence minimum
- **Market Analysis**: 80% confidence minimum
- **General Queries**: 60% confidence minimum

## 🔧 Integration Methods

### Method 1: MCP Protocol (Recommended)

```python
import mcp

# Initialize MCP client for Tyra
client = mcp.Client("tyra-memory-server")

# Create Tyra session
session = await client.call("create_session", {
    "agent_id": "tyra",
    "user_id": "trader_001",
    "metadata": {
        "trading_session": True,
        "risk_level": "moderate",
        "account_id": "ACCT123"
    }
})

# Store trading memory with high-confidence requirement
result = await client.call("save_memory", {
    "text": "AAPL showing strong bullish momentum with RSI at 65, volume above average",
    "agent_id": "tyra",
    "session_id": session["session_id"],
    "metadata": {
        "symbol": "AAPL",
        "indicator": "RSI",
        "signal": "bullish",
        "confidence": 87
    },
    "extract_entities": True
})

# Search for trading insights
insights = await client.call("search_memories", {
    "query": "AAPL bullish signals RSI volume",
    "agent_id": "tyra",
    "session_id": session["session_id"],
    "min_confidence": 0.8,  # High confidence for trading
    "include_analysis": True
})
```

### Method 2: Memory Client Library

```python
from src.clients.memory_client import MemoryClient

# Initialize memory client
memory_client = MemoryClient(
    base_url="http://localhost:8000",
    agent_id="tyra",
    api_key="your-api-key"
)

# Initialize client
await memory_client.initialize()

# Store market analysis
result = await memory_client.store_memory(
    content="SPY broke above 450 resistance with strong volume confirmation",
    metadata={
        "symbol": "SPY",
        "action": "breakout",
        "level": 450,
        "confirmation": "volume",
        "timestamp": "2024-01-15T09:30:00Z"
    },
    extract_entities=True
)

# Search for similar patterns
similar_patterns = await memory_client.search_memories(
    query="SPY breakout resistance volume",
    min_confidence=0.85,
    include_confidence_analysis=True
)

# Verify confidence before trading decision
if similar_patterns["confidence_analysis"]["overall_confidence"] >= 95:
    print("High confidence - suitable for trading decision")
else:
    print("Low confidence - manual review required")
```

### Method 3: REST API

```bash
# Store trading signal
curl -X POST http://localhost:8000/v1/memory/store \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "content": "Bitcoin forming ascending triangle pattern, expecting breakout above 45000",
    "agent_id": "tyra",
    "metadata": {
      "symbol": "BTC",
      "pattern": "ascending_triangle",
      "target": 45000,
      "timeframe": "4h"
    },
    "extract_entities": true
  }'

# Search for trading patterns
curl -X POST http://localhost:8000/v1/memory/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "Bitcoin ascending triangle breakout",
    "agent_id": "tyra",
    "min_confidence": 0.9,
    "include_analysis": true,
    "rerank": true
  }'
```

## 📊 Trading-Specific Features

### Risk Management Integration

```python
# Example: Risk-aware memory storage
async def store_trading_signal(signal_data):
    # Calculate risk score
    risk_score = calculate_risk_score(signal_data)

    # Adjust confidence requirements based on risk
    if risk_score > 0.7:  # High risk
        min_confidence = 0.95
    elif risk_score > 0.4:  # Medium risk
        min_confidence = 0.85
    else:  # Low risk
        min_confidence = 0.75

    # Store with risk metadata
    result = await memory_client.store_memory(
        content=signal_data["description"],
        metadata={
            **signal_data,
            "risk_score": risk_score,
            "min_confidence_required": min_confidence,
            "risk_category": get_risk_category(risk_score)
        }
    )

    return result
```

### Market Sentiment Analysis

```python
# Store sentiment data with temporal context
async def store_market_sentiment(sentiment_data):
    result = await memory_client.store_memory(
        content=f"Market sentiment: {sentiment_data['sentiment']} "
                f"based on {sentiment_data['indicators']}",
        metadata={
            "sentiment_score": sentiment_data["score"],
            "market_phase": sentiment_data["phase"],
            "volatility": sentiment_data["volatility"],
            "timestamp": sentiment_data["timestamp"],
            "data_sources": sentiment_data["sources"]
        },
        extract_entities=True,
        chunk_content=False  # Keep sentiment data together
    )

    return result

# Search for sentiment patterns
async def get_sentiment_history(symbol, timeframe="1d"):
    results = await memory_client.search_memories(
        query=f"{symbol} market sentiment {timeframe}",
        search_type="temporal",
        time_range={
            "start": (datetime.now() - timedelta(days=30)).isoformat(),
            "end": datetime.now().isoformat()
        },
        min_confidence=0.7
    )

    return results
```

## 🎯 Advanced RAG for Trading

### Context-Aware Query Enhancement

```python
async def enhanced_trading_query(query, context=None):
    """
    Enhanced query processing for trading-specific context.
    """
    # Add trading context to query
    enhanced_query = f"{query}"

    if context:
        if context.get("symbol"):
            enhanced_query += f" symbol:{context['symbol']}"
        if context.get("timeframe"):
            enhanced_query += f" timeframe:{context['timeframe']}"
        if context.get("strategy"):
            enhanced_query += f" strategy:{context['strategy']}"

    # Search with enhanced query
    results = await memory_client.search_memories(
        query=enhanced_query,
        agent_id="tyra",
        min_confidence=0.8,
        rerank=True,
        include_analysis=True
    )

    # Validate for trading confidence
    if results["confidence_analysis"]["overall_confidence"] >= 95:
        results["trading_suitable"] = True
        results["confidence_level"] = "rock_solid"
    else:
        results["trading_suitable"] = False
        results["confidence_level"] = "review_required"

    return results
```

### Hallucination Detection for Trading

```python
async def validate_trading_insight(insight, supporting_data):
    """
    Validate trading insights against hallucination.
    """
    # Analyze insight for hallucination
    analysis = await memory_client.analyze_response(
        response=insight,
        query="trading analysis validation",
        retrieved_memories=supporting_data
    )

    # Trading-specific validation
    if analysis["analysis"]["is_hallucination"]:
        return {
            "valid": False,
            "reason": "hallucination_detected",
            "confidence": analysis["analysis"]["overall_confidence"],
            "recommendation": "manual_review_required"
        }

    if analysis["analysis"]["overall_confidence"] < 95:
        return {
            "valid": False,
            "reason": "low_confidence",
            "confidence": analysis["analysis"]["overall_confidence"],
            "recommendation": "additional_data_required"
        }

    return {
        "valid": True,
        "confidence": analysis["analysis"]["overall_confidence"],
        "grounding_score": analysis["analysis"]["grounding_score"]
    }
```

## 🔗 External System Integration

### Trading Platform Webhooks

```python
# Example: TradingView webhook integration
@app.post("/webhooks/tradingview")
async def handle_tradingview_signal(signal_data: dict):
    """Handle TradingView webhook signals."""

    # Store signal in Tyra memory
    memory_result = await memory_client.store_memory(
        content=f"TradingView signal: {signal_data['message']}",
        metadata={
            "source": "tradingview",
            "symbol": signal_data["ticker"],
            "signal": signal_data["strategy"]["order_action"],
            "price": signal_data["strategy"]["order_price"],
            "timestamp": signal_data["time"]
        },
        agent_id="tyra"
    )

    # Search for similar signals
    similar_signals = await memory_client.search_memories(
        query=f"{signal_data['ticker']} {signal_data['strategy']['order_action']}",
        agent_id="tyra",
        min_confidence=0.8
    )

    # Return enhanced signal with historical context
    return {
        "signal_stored": memory_result["success"],
        "memory_id": memory_result.get("memory_id"),
        "similar_signals": len(similar_signals.get("results", [])),
        "confidence_score": similar_signals.get("confidence_analysis", {}).get("overall_confidence", 0)
    }
```

### Risk Management Alerts

```python
async def setup_risk_monitoring():
    """Setup automated risk monitoring with memory system."""

    # Monitor for high-risk patterns
    risk_patterns = await memory_client.search_memories(
        query="high risk drawdown loss volatility",
        agent_id="tyra",
        min_confidence=0.9,
        top_k=20
    )

    # Create risk alerts based on historical patterns
    if risk_patterns["confidence_analysis"]["overall_confidence"] > 0.95:
        # Set up monitoring for similar conditions
        await setup_pattern_monitoring(risk_patterns["results"])

    return risk_patterns
```

## 📈 Performance Optimization

### Trading-Specific Caching

```python
# Configure cache for trading data
trading_cache_config = {
    "market_data": {"ttl": 300},      # 5 minutes for market data
    "signals": {"ttl": 1800},         # 30 minutes for signals
    "analysis": {"ttl": 3600},        # 1 hour for analysis
    "risk_metrics": {"ttl": 900}      # 15 minutes for risk data
}

# Use trading-optimized search
async def fast_signal_search(symbol, signal_type):
    """Optimized search for trading signals."""
    cache_key = f"signal:{symbol}:{signal_type}"

    # Check cache first
    cached_result = await cache.get(cache_key)
    if cached_result:
        return cached_result

    # Search memory system
    results = await memory_client.search_memories(
        query=f"{symbol} {signal_type}",
        agent_id="tyra",
        top_k=5,
        search_type="vector",  # Faster than hybrid for known patterns
        min_confidence=0.8
    )

    # Cache results
    await cache.set(cache_key, results, ttl=trading_cache_config["signals"]["ttl"])

    return results
```

## 🚨 Safety and Compliance

### Trading Compliance

```python
async def compliance_check(trading_decision):
    """
    Ensure trading decisions meet compliance requirements.
    """
    # Check confidence requirements
    if trading_decision["confidence"] < 95:
        return {
            "approved": False,
            "reason": "insufficient_confidence",
            "required_confidence": 95,
            "actual_confidence": trading_decision["confidence"]
        }

    # Check for hallucination
    if trading_decision.get("hallucination_detected"):
        return {
            "approved": False,
            "reason": "hallucination_detected",
            "grounding_score": trading_decision.get("grounding_score", 0)
        }

    # Verify against historical data
    verification = await verify_against_history(trading_decision)

    return {
        "approved": verification["consistent"],
        "confidence": trading_decision["confidence"],
        "verification_score": verification["score"]
    }
```

### Audit Trail

```python
async def log_trading_decision(decision_data):
    """
    Log trading decisions with full audit trail.
    """
    audit_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "agent_id": "tyra",
        "decision_type": "trading",
        "symbol": decision_data["symbol"],
        "action": decision_data["action"],
        "confidence": decision_data["confidence"],
        "supporting_memories": decision_data["memory_ids"],
        "risk_score": decision_data["risk_score"],
        "compliance_check": decision_data["compliance_status"]
    }

    # Store in memory system for audit
    await memory_client.store_memory(
        content=f"Trading decision: {decision_data['action']} {decision_data['symbol']}",
        metadata=audit_entry,
        agent_id="tyra"
    )

    return audit_entry
```

## 📚 Best Practices

### 1. Confidence Management
- Always check confidence scores before trading decisions
- Use 95% minimum confidence for actual trades
- Implement manual review for sub-threshold decisions

### 2. Risk-Aware Memory Storage
- Include risk scores in all trading memories
- Tag memories with risk categories
- Implement risk-based retention policies

### 3. Temporal Awareness
- Include timestamp metadata in all market data
- Use temporal search for time-sensitive analysis
- Implement market hours awareness

### 4. Data Quality
- Validate all market data before storage
- Implement source attribution
- Use entity extraction for structured data

### 5. Performance Monitoring
- Monitor query latencies for trading-critical searches
- Track confidence score distributions
- Alert on degraded performance

## 🔧 Troubleshooting

### Common Issues

**Low Confidence Scores**
```python
# Increase data quality and quantity
await memory_client.store_memory(
    content=detailed_analysis,
    metadata=comprehensive_metadata,
    extract_entities=True,
    chunk_content=False  # Keep related data together
)
```

**Slow Query Performance**
```python
# Use optimized search parameters
results = await memory_client.search_memories(
    query=query,
    search_type="vector",  # Faster than hybrid
    top_k=5,               # Limit results
    min_confidence=0.9     # Higher threshold
)
```

**Memory Isolation Issues**
```python
# Ensure proper session management
session = await create_agent_session(
    agent_id="tyra",
    user_id=user_id,
    metadata={"isolation_level": "strict"}
)
```

## 📞 Support

For Tyra-specific integration issues:

1. **Check Logs**: Look for agent-specific log entries with Tyra context
2. **Verify Configuration**: Ensure `config/agents.yaml` has correct Tyra settings
3. **Test Confidence**: Use the confidence testing endpoints
4. **Monitor Performance**: Check trading-specific metrics

---

🎯 **Tyra Integration Complete!** Your trading agent now has access to advanced memory capabilities with high-confidence requirements and risk-aware features.
