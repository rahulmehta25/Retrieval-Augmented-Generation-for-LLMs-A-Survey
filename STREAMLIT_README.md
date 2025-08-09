# 🎯 Streamlit UI for RAG System

A modern, interactive web interface for experimenting with the RAG (Retrieval-Augmented Generation) system, complete with RAGAS evaluation metrics and beautiful visualizations.

## 🌟 Features

### 1. **Interactive Chat Interface**
- Clean, user-friendly chat interface
- Real-time question answering
- Display of retrieved contexts
- Query time tracking

### 2. **Multiple RAG Implementations**
- **Naive RAG**: Basic retrieve-then-read approach
- **Advanced RAG**: With query optimization techniques
- **Modular RAG**: Factory pattern with advanced features

### 3. **RAGAS Evaluation Metrics**
- Faithfulness score
- Answer relevancy
- Context relevancy
- Context precision
- Overall performance score
- Visual metric displays with color coding

### 4. **Document Management**
- Upload multiple document formats (TXT, PDF, MD, DOCX, XLSX)
- Real-time document indexing
- Track uploaded documents
- Support for batch processing

### 5. **Evaluation Dashboard**
- Interactive charts with Plotly
- Metrics trend over time
- Score distribution analysis
- Radar chart for performance overview
- Export results as JSON or CSV

### 6. **Benchmark Testing**
- Run benchmarks on standard datasets
- Support for SQuAD, MS MARCO
- Custom dataset evaluation
- Performance metrics and latency stats

## 🚀 Quick Start

### Option 1: Using the Run Script (Recommended)

```bash
# Make the script executable (first time only)
chmod +x run_streamlit.sh

# Run the Streamlit app
./run_streamlit.sh
```

The script will:
- Create/activate a virtual environment
- Install all dependencies
- Check for Ollama and required models
- Start the Streamlit server

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit
streamlit run streamlit_app.py
```

### Option 3: Quick Run (if dependencies are installed)

```bash
streamlit run streamlit_app.py
```

## 📱 Using the Interface

### Step 1: Initialize the System

1. Open sidebar (left side)
2. Select RAG type:
   - **Naive**: Fast, simple
   - **Advanced**: Better for complex queries
   - **Modular**: Most features
3. Enable/disable RAGAS evaluation
4. Click "🚀 Initialize System"

### Step 2: Upload Documents

1. In sidebar, find "Document Management"
2. Click "Browse files" or drag & drop
3. Select your documents
4. Click "📥 Index Documents"
5. Wait for indexing to complete

### Step 3: Ask Questions

1. Go to the "💬 Chat" tab
2. Type your question
3. Click "🔍 Search"
4. View:
   - Answer
   - RAGAS metrics (if enabled)
   - Retrieved contexts (expandable)
   - Query time

### Step 4: Analyze Performance

1. Go to "📊 Evaluation" tab
2. View:
   - Average metrics
   - Trends over time
   - Score distribution
   - Latest query radar chart
3. Export results as needed

## 🎨 Interface Components

### Sidebar
- **Configuration**: RAG type and evaluation settings
- **System Status**: Current system state
- **Document Management**: Upload and track documents
- **Data Management**: Clear history and results

### Main Tabs

#### 💬 Chat Tab
- Question input field
- Answer display with success formatting
- RAGAS metrics display (color-coded)
- Expandable context viewer
- Chat history (last 5 queries)

#### 📊 Evaluation Tab
- Average metrics cards
- Line chart: Metrics over time
- Box plot: Score distribution
- Radar chart: Latest query performance
- Detailed results table
- Export buttons (JSON/CSV)

#### 📈 Benchmark Tab
- Dataset selection
- Max examples setting
- Run benchmark button
- Results display
- Performance metrics

#### 📚 Documentation Tab
- Quick start guide
- RAG types explanation
- Metrics descriptions
- Tips and troubleshooting

## 🎯 Best Practices

### For Better Answers
1. **Upload relevant documents** - The system only knows what you upload
2. **Be specific** - Clear questions get better answers
3. **Check contexts** - Verify the source of answers
4. **Use appropriate RAG type**:
   - Naive: General queries
   - Advanced: Complex questions
   - Modular: Production use

### For Evaluation
1. **Enable RAGAS** for quality metrics
2. **Monitor trends** to spot issues
3. **Export results** for analysis
4. **Compare RAG types** for your use case

### Performance Tips
1. **Index documents once** - Reuse for multiple queries
2. **Batch similar questions** - Better context reuse
3. **Clear old results** - Keep interface responsive
4. **Use smaller documents** - Faster processing

## 🔧 Configuration

### Modify `config.yaml` for:

```yaml
# Embedding model
embedder:
  model_name: all-MiniLM-L6-v2  # Change model here

# LLM settings
generator:
  model_name: gemma:2b  # Change to llama2, mistral, etc.
  temperature: 0.7      # Adjust creativity

# Retrieval settings
vector_store:
  collection_name: rag_documents
  top_k: 5  # Number of contexts to retrieve
```

### Environment Variables

```bash
# For OpenAI (optional)
export OPENAI_API_KEY="your-key-here"

# For custom Ollama host
export OLLAMA_HOST="http://localhost:11434"
```

## 🐛 Troubleshooting

### Common Issues

1. **"System not initialized"**
   - Click "Initialize System" in sidebar
   - Check for error messages
   - Verify config.yaml exists

2. **"No documents loaded"**
   - Upload documents first
   - Wait for indexing to complete
   - Check file formats are supported

3. **Low RAGAS scores**
   - Try different RAG type
   - Improve document quality
   - Rephrase questions

4. **Slow performance**
   - Reduce number of documents
   - Use smaller chunks
   - Check Ollama is running

5. **Import errors**
   ```bash
   pip install -r requirements.txt
   ```

### Debug Mode

Add to your command:
```bash
streamlit run streamlit_app.py --logger.level=debug
```

## 📊 Metrics Explained

### Visual Indicators
- 🟢 **Green** (≥0.8): Excellent
- 🟡 **Yellow** (0.6-0.8): Good
- 🔴 **Red** (<0.6): Needs improvement

### Metric Meanings
- **Faithfulness**: Is the answer grounded in retrieved documents?
- **Answer Relevancy**: Does the answer address the question?
- **Context Relevancy**: Are retrieved documents relevant?
- **Context Precision**: Are most relevant docs ranked first?
- **Overall**: Average of all metrics

## 🎉 Tips & Tricks

1. **Keyboard Shortcuts**
   - `Ctrl+Enter`: Submit question (in text field)
   - `R`: Rerun app
   - `C`: Clear cache

2. **URL Parameters**
   ```
   http://localhost:8501/?question=What%20is%20RAG
   ```

3. **Custom Themes**
   Edit `.streamlit/config.toml`:
   ```toml
   [theme]
   primaryColor = "#667eea"
   backgroundColor = "#FFFFFF"
   ```

4. **Share Results**
   - Screenshot the evaluation dashboard
   - Export metrics as CSV for reports
   - Save JSON for programmatic analysis

## 🔗 Related Documentation

- [Main README](README.md)
- [RAGAS Documentation](RAGAS_DOCUMENTATION.md)
- [API Documentation](api_server.py)
- [RAG Survey Paper](https://arxiv.org/abs/2312.10997)

## 📝 License

MIT License - Feel free to use and modify!

---

**Enjoy exploring RAG with the Streamlit interface! 🚀**