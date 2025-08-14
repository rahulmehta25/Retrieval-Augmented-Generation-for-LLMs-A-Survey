# 🧪 RAG System Testing Guide

## ✅ System Status
- **API Server**: Running at http://127.0.0.1:8090 ✅
- **Streamlit UI**: Running at http://127.0.0.1:8501 ✅
- **Documents Loaded**: 4 documents in system
- **Models Available**: gemma:2b, llama2:7b, mistral:7b, qwen2.5-coder:7b

## 📋 Manual Testing Checklist

### 1. **Streamlit Dashboard Tests** (http://127.0.0.1:8501)

#### 📄 Documents Tab
- [ ] Upload `test_document.txt`
- [ ] Click "Process" button
- [ ] Verify document appears in list
- [ ] Try uploading different formats (PDF, DOCX if available)

#### 🤖 RAG Testing Tab
Test each RAG type with the query: **"What is Machine Learning?"**

- [ ] **Naive RAG** 
  - Expected: Basic answer with sources
  - RAGAS scores should appear if evaluation enabled
  
- [ ] **Advanced RAG**
  - Expected: Optimized query processing
  - May show query rewriting in logs
  
- [ ] **Modular RAG**
  - Expected: Component-based processing
  - Should return answer with contexts
  
- [ ] **Graph RAG**
  - First time: May need to build graph
  - Expected: Knowledge graph-based retrieval
  
- [ ] **Self RAG**
  - Expected: Self-reflection in processing
  - May take longer due to critique loop
  
- [ ] **Auto**
  - Expected: Automatic routing to best strategy
  - Should fallback gracefully if advanced features unavailable

#### ⚡ Streaming Tab
- [ ] Enter "What is AI?"
- [ ] Click "Test Streaming"
- [ ] Should see progressive text display

#### 🛡️ Error Handling Tab
- [ ] Test Empty Query - Should handle gracefully
- [ ] Test Very Long Query - Should limit or handle
- [ ] Test Special Characters - Should process safely
- [ ] Test No Documents - Should provide appropriate message

#### 📈 Analytics Tab
- [ ] View query distribution pie chart
- [ ] Check processing time box plot
- [ ] Review average RAGAS scores (if evaluations run)

#### 💬 History Tab
- [ ] Verify all queries appear
- [ ] Check timestamps are correct
- [ ] Test "Export Chat History" button

### 2. **API Endpoint Tests** (Using curl)

```bash
# Test 1: Basic Query
curl -X POST http://127.0.0.1:8090/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Deep Learning?", "stream": false}'

# Test 2: Advanced RAG
curl -X POST http://127.0.0.1:8090/api/chat/query/advanced \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain neural networks", "stream": false}'

# Test 3: Evaluation
curl -X POST http://127.0.0.1:8090/api/evaluate/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is NLP?"}'

# Test 4: Auto-routing
curl -X POST http://127.0.0.1:8090/api/chat/query/auto \
  -H "Content-Type: application/json" \
  -d '{"question": "How does RAG work?", "stream": false}'
```

### 3. **Performance Tests**

#### Response Time Check
Compare response times for the same query across different RAG types:
1. Naive RAG: Should be fastest (~1-2s)
2. Advanced RAG: Slightly slower due to optimization (~2-3s)
3. Self RAG: Slowest due to reflection (~3-5s)

#### RAGAS Score Comparison
Enable evaluation and compare scores:
- Faithfulness: How grounded is the answer?
- Answer Relevancy: Does it answer the question?
- Context Relevancy: Are retrieved contexts good?
- Context Precision: Is ranking optimal?

## 🎯 Expected Results

### ✅ Working Features
1. **Document Upload** - Multiple formats supported
2. **Basic RAG Query** - Retrieve and generate answers
3. **RAGAS Evaluation** - Quality metrics for responses
4. **Multiple RAG Paradigms** - Different strategies available
5. **Analytics Dashboard** - Performance tracking
6. **Chat History** - Session persistence

### ⚠️ Features That May Need Setup
1. **Graph RAG** - Requires building knowledge graph first
2. **Streaming** - Depends on SSE support
3. **Advanced RAG** - May need specific model configurations

### 🔧 Troubleshooting

**If queries return empty/poor answers:**
- Upload more documents
- Check if Ollama is running
- Verify model is loaded (gemma:2b)

**If evaluation fails:**
- Evaluation requires additional LLM calls
- May timeout with slow models
- Try with smaller queries first

**If advanced RAG types fail:**
- Check API logs for initialization errors
- May fallback to naive RAG automatically
- Verify all dependencies installed

## 📊 Current System Metrics

```json
{
  "documents_loaded": 4,
  "model": "gemma:2b",
  "rag_types_available": 6,
  "evaluation_enabled": true,
  "streaming_available": true,
  "api_status": "healthy"
}
```

## 🎉 Success Criteria

The system is working correctly if:
- [ ] Can upload and process documents
- [ ] Can query with at least naive RAG
- [ ] Returns relevant answers to questions
- [ ] Shows RAGAS evaluation scores
- [ ] Tracks history and analytics
- [ ] Handles errors gracefully

## 📝 Notes

- The system is designed to degrade gracefully
- If advanced features fail, it falls back to basic RAG
- All errors are logged and shown to user
- The dashboard shows which components are available

---

**Last tested**: Current session
**Status**: ✅ OPERATIONAL
**Documents**: 4 loaded
**Primary model**: gemma:2b