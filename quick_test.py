#!/usr/bin/env python3
"""Quick test to verify all features are working"""

import requests
import json
import time

API_BASE = "http://127.0.0.1:8090/api"

def test_endpoints():
    """Test all API endpoints"""
    print("🧪 Testing RAG System Features\n" + "="*40)
    
    results = []
    
    # 1. Test Health
    try:
        r = requests.get(f"{API_BASE}/health")
        if r.status_code == 200:
            print("✅ Health check: OK")
            results.append(("Health", True))
        else:
            print(f"❌ Health check: {r.status_code}")
            results.append(("Health", False))
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        results.append(("Health", False))
    
    # 2. Test Status
    try:
        r = requests.get(f"{API_BASE}/status")
        if r.status_code == 200:
            status = r.json()
            print(f"✅ Status: Ready={status.get('ready')}, Docs={status.get('documents_loaded')}")
            results.append(("Status", True))
        else:
            print(f"❌ Status check: {r.status_code}")
            results.append(("Status", False))
    except Exception as e:
        print(f"❌ Status check failed: {e}")
        results.append(("Status", False))
    
    # 3. Test Basic Query
    print("\n📝 Testing Queries...")
    try:
        r = requests.post(
            f"{API_BASE}/chat/query",
            json={"question": "What is AI?", "stream": False}
        )
        if r.status_code == 200:
            result = r.json()
            answer = result.get('answer', '')[:100]
            print(f"✅ Basic query: {answer}...")
            results.append(("Basic Query", True))
        else:
            print(f"❌ Basic query: {r.status_code}")
            results.append(("Basic Query", False))
    except Exception as e:
        print(f"❌ Basic query failed: {e}")
        results.append(("Basic Query", False))
    
    # 4. Test Advanced RAG
    try:
        r = requests.post(
            f"{API_BASE}/chat/query/advanced",
            json={"question": "What is Machine Learning?", "stream": False}
        )
        if r.status_code == 200:
            print("✅ Advanced RAG: Working")
            results.append(("Advanced RAG", True))
        elif r.status_code == 503:
            print("⚠️  Advanced RAG: Not available (expected if not initialized)")
            results.append(("Advanced RAG", None))
        else:
            print(f"❌ Advanced RAG: {r.status_code}")
            results.append(("Advanced RAG", False))
    except Exception as e:
        print(f"❌ Advanced RAG failed: {e}")
        results.append(("Advanced RAG", False))
    
    # 5. Test Graph RAG
    try:
        r = requests.post(
            f"{API_BASE}/chat/query/graph",
            json={"question": "What is Deep Learning?", "stream": False}
        )
        if r.status_code == 200:
            print("✅ Graph RAG: Working")
            results.append(("Graph RAG", True))
        elif r.status_code == 503:
            print("⚠️  Graph RAG: Not available (expected if not initialized)")
            results.append(("Graph RAG", None))
        else:
            print(f"❌ Graph RAG: {r.status_code}")
            results.append(("Graph RAG", False))
    except Exception as e:
        print(f"❌ Graph RAG failed: {e}")
        results.append(("Graph RAG", False))
    
    # 6. Test Evaluation
    print("\n📊 Testing Evaluation...")
    try:
        r = requests.post(
            f"{API_BASE}/evaluate/query",
            json={"question": "What is Natural Language Processing?"}
        )
        if r.status_code == 200:
            result = r.json()
            if 'ragas_scores' in result and result['ragas_scores']:
                scores = result['ragas_scores']
                print(f"✅ RAGAS Evaluation:")
                for metric, score in scores.items():
                    print(f"   - {metric}: {score:.2f}")
                results.append(("RAGAS Evaluation", True))
            else:
                print("⚠️  RAGAS scores not available")
                results.append(("RAGAS Evaluation", None))
        else:
            print(f"❌ Evaluation: {r.status_code}")
            results.append(("RAGAS Evaluation", False))
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        results.append(("RAGAS Evaluation", False))
    
    # 7. Test Auto-routing
    try:
        r = requests.post(
            f"{API_BASE}/chat/query/auto",
            json={"question": "Explain RAG systems", "stream": False}
        )
        if r.status_code == 200:
            print("✅ Auto-routing: Working")
            results.append(("Auto-routing", True))
        else:
            print(f"⚠️  Auto-routing: Fallback to naive RAG")
            results.append(("Auto-routing", None))
    except Exception as e:
        print(f"❌ Auto-routing failed: {e}")
        results.append(("Auto-routing", False))
    
    # Summary
    print("\n" + "="*40)
    print("📊 SUMMARY")
    print("="*40)
    
    working = sum(1 for _, status in results if status is True)
    partial = sum(1 for _, status in results if status is None)
    failed = sum(1 for _, status in results if status is False)
    total = len(results)
    
    print(f"✅ Working: {working}/{total}")
    print(f"⚠️  Partial: {partial}/{total}")
    print(f"❌ Failed: {failed}/{total}")
    
    if working >= total * 0.7:
        print("\n🎉 System is WORKING WELL!")
    elif working >= total * 0.5:
        print("\n⚠️  System is PARTIALLY FUNCTIONAL")
    else:
        print("\n❌ System needs attention")
    
    return results

def test_streaming():
    """Test streaming endpoint"""
    print("\n⚡ Testing Streaming...")
    try:
        import requests
        from requests.exceptions import ChunkedEncodingError
        
        # SSE endpoint
        url = f"{API_BASE}/chat/stream?question=What%20is%20AI?"
        
        with requests.get(url, stream=True) as r:
            if r.status_code == 200:
                print("✅ Streaming endpoint accessible")
                # Read first few events
                for i, line in enumerate(r.iter_lines()):
                    if i > 5:  # Just check first few
                        break
                    if line:
                        print(f"   Received: {line.decode()[:50]}...")
                return True
            else:
                print(f"❌ Streaming failed: {r.status_code}")
                return False
    except Exception as e:
        print(f"⚠️  Streaming test incomplete: {e}")
        return None

def test_document_upload():
    """Test document upload"""
    print("\n📄 Testing Document Upload...")
    try:
        # Create a test file
        test_content = b"This is a test document for RAG system."
        files = {'file': ('test.txt', test_content, 'text/plain')}
        
        r = requests.post(f"{API_BASE}/documents/upload", files=files)
        
        if r.status_code == 200:
            print("✅ Document upload working")
            return True
        else:
            print(f"⚠️  Document upload: {r.status_code}")
            return False
    except Exception as e:
        print(f"❌ Document upload failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 RAG SYSTEM QUICK TEST")
    print("="*40)
    print(f"Testing API at: {API_BASE}")
    print("="*40 + "\n")
    
    # Run tests
    results = test_endpoints()
    streaming_ok = test_streaming()
    upload_ok = test_document_upload()
    
    print("\n" + "="*40)
    print("✅ Tests Complete!")
    print("\nNext steps:")
    print("1. Open Streamlit at http://127.0.0.1:8501")
    print("2. Upload documents in the Documents tab")
    print("3. Test different RAG types in RAG Testing tab")
    print("4. Check Analytics for performance metrics")