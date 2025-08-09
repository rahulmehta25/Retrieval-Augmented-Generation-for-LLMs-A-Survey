#!/usr/bin/env python3
"""
Verify RAG Chunk Precision
Tests that the system retrieves ONLY relevant chunks and not unrelated content
"""

import requests
import json

API_URL = "http://localhost:8090"

def test_precision():
    """Test that queries return only relevant chunks"""
    
    tests = [
        {
            "query": "What is XGBoost?",
            "should_find": ["machine_learning.txt"],
            "should_not_find": ["python_basics.txt", "web_development.md", "BeachBox"]
        },
        {
            "query": "What are tuples in Python?",
            "should_find": ["python_basics.txt"],
            "should_not_find": ["machine_learning.txt", "web_development.md", "BeachBox"]
        },
        {
            "query": "What is Flexbox?",
            "should_find": ["web_development.md"],
            "should_not_find": ["python_basics.txt", "machine_learning.txt", "BeachBox"]
        },
        {
            "query": "Tell me about BeachBox",
            "should_find": ["BeachBox"],
            "should_not_find": ["python_basics.txt", "machine_learning.txt", "web_development.md"]
        }
    ]
    
    print("CHUNK PRECISION TEST")
    print("=" * 60)
    print("Verifying that only relevant documents are retrieved\n")
    
    all_passed = True
    
    for test in tests:
        print(f"Query: '{test['query']}'")
        
        response = requests.post(
            f"{API_URL}/api/chat/query",
            json={"question": test["query"]}
        )
        result = response.json()
        sources = result.get("sources", [])
        
        # Check sources
        sources_str = " ".join(sources).lower()
        
        # Check if expected documents are found
        found_expected = any(doc.lower() in sources_str for doc in test["should_find"])
        
        # Check if unwanted documents are NOT found
        found_unwanted = any(doc.lower() in sources_str for doc in test["should_not_find"])
        
        passed = found_expected and not found_unwanted
        all_passed = all_passed and passed
        
        print(f"  Sources: {sources[:2] if len(sources) > 2 else sources}")
        print(f"  Expected docs found: {found_expected}")
        print(f"  No unwanted docs: {not found_unwanted}")
        print(f"  Status: {'✅ PASS' if passed else '❌ FAIL'}")
        print()
    
    print("-" * 60)
    print(f"Overall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    print("\nConclusion:")
    if all_passed:
        print("The RAG system is correctly retrieving relevant chunks")
        print("and filtering out unrelated documents!")
    else:
        print("The RAG system may need tuning for better precision.")

if __name__ == "__main__":
    test_precision()