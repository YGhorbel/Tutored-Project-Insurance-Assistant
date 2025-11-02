"""End-to-end pipeline test"""
import requests
import time
import sys


def test_api_health():
    """Test API health endpoint"""
    print("Testing API health...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        response.raise_for_status()
        data = response.json()
        print(f"✓ API is healthy: {data}")
        return data.get('index_loaded', False)
    except Exception as e:
        print(f"✗ API health check failed: {e}")
        return False


def test_query(question: str):
    """Test RAG query"""
    print(f"\nQuerying: {question}")
    try:
        response = requests.post(
            "http://localhost:8000/query",
            json={"question": question, "top_k": 3},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Query successful")
        print(f"  Answer: {data['answer'][:200]}...")
        print(f"  Chunks found: {len(data['chunks'])}")
        print(f"  Processing time: {data['processing_time_ms']:.2f}ms")
        
        return True
    except Exception as e:
        print(f"✗ Query failed: {e}")
        return False


def main():
    """Run pipeline tests"""
    print("=== Insurance RAG Pipeline Test ===\n")
    
    # Test API health
    if not test_api_health():
        print("\nAPI not ready. Please ensure the system is running.")
        sys.exit(1)
    
    # Wait a bit
    time.sleep(2)
    
    # Test queries
    test_questions = [
        "What are the solvency requirements for insurance companies?",
        "What are the consumer protection regulations?",
        "What are the reporting obligations for insurers?"
    ]
    
    success_count = 0
    for question in test_questions:
        if test_query(question):
            success_count += 1
        time.sleep(1)
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {success_count}/{len(test_questions)}")
    
    if success_count == len(test_questions):
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
