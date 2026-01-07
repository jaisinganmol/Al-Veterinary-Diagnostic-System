"""
Quick System Test Script

Tests the complete veterinary voice-AI system to verify:
- Configuration
- RAG system
- Multi-agent orchestration
- Performance targets
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.app.config import settings, validate_config
from backend.app.graph.orchestrator import get_orchestrator
from backend.app.rag.pinecone_retriever import get_rag_system


async def test_configuration():
    """Test 1: Configuration"""
    print("\n" + "=" * 70)
    print("TEST 1: Configuration")
    print("=" * 70)

    try:
        validate_config()
        print("Configuration valid")
        print(f"  - Claude Model: {settings.CLAUDE_MODEL}")
        print(f"  - Pinecone Index: {settings.PINECONE_INDEX_NAME}")
        print(f"  - RAG Threshold: {settings.RAG_HIGH_CONFIDENCE_THRESHOLD}")
        return True
    except Exception as e:
        print(f"Configuration error: {e}")
        return False


async def test_rag_system():
    """Test 2: RAG System"""
    print("\n" + "=" * 70)
    print("TEST 2: RAG System (Pinecone)")
    print("=" * 70)

    try:
        rag = get_rag_system()

        # Test search
        print("Testing RAG search...")
        result = await rag.search(
            query="Cerenia dosing for dogs",
            top_k=3
        )

        print(f"RAG system operational")
        print(f"  - Documents retrieved: {len(result['documents'])}")
        print(f"  - Confidence: {result['confidence']:.2%}")
        print(f"  - Web search needed: {result['requires_web_search']}")

        if result['confidence'] < 0.60:
            print("Warning: Low RAG confidence. Consider adding more knowledge.")

        return True

    except Exception as e:
        print(f"RAG error: {e}")
        return False


async def test_orchestrator():
    """Test 3: Multi-Agent Orchestrator"""
    print("\n" + "=" * 70)
    print("TEST 3: Multi-Agent Orchestrator (LangGraph)")
    print("=" * 70)

    try:
        orchestrator = get_orchestrator()

        # Test query
        test_query = "Emergency triage: 25kg dog vomiting blood"

        print(f"Testing query: '{test_query}'")
        print("Processing...")

        result = await orchestrator.process(
            query=test_query,
            species="canine",
            weight_kg=25,
            age_years=5
        )

        print(f"\nOrchestrator operational")
        print(f"  - Agents used: {', '.join(result['agents_used'])}")
        print(f"  - Intent: {result['intent']}")
        print(f"  - Urgency: {result['urgency']}")
        print(f"  - Latency: {result['latency_ms']}ms")
        print(f"  - RAG confidence: {result['rag_confidence']:.2%}")
        print(f"  - Hallucination risk: {result['hallucination_risk']:.2%}")
        print(f"  - Grounding: {result['grounding_quality']}")

        # Check performance targets
        print("\nPerformance Targets:")

        # Sub-2s latency
        if result['latency_ms'] <= 2000:
            print(f"  Sub-2s latency: {result['latency_ms']}ms")
        else:
            print(f"  Latency exceeded: {result['latency_ms']}ms (target: <2000ms)")

        # 85% hallucination reduction
        if result['hallucination_risk'] <= 0.15:
            print(f"  Hallucination risk: {result['hallucination_risk']:.2%} (target: <15%)")
        else:
            print(f"  Hallucination risk: {result['hallucination_risk']:.2%} (target: <15%)")

        # High RAG confidence
        if result['rag_confidence'] >= 0.80:
            print(f"  RAG confidence: {result['rag_confidence']:.2%} (target: >80%)")
        else:
            print(f"  RAG confidence: {result['rag_confidence']:.2%} (target: >80%)")

        # Show response preview
        print(f"\nResponse Preview:")
        print(result['answer'][:300] + "...")

        return True

    except Exception as e:
        print(f"Orchestrator error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_graph_visualization():
    """Test 4: Graph Visualization"""
    print("\n" + "=" * 70)
    print("TEST 4: Graph Visualization")
    print("=" * 70)

    try:
        orchestrator = get_orchestrator()
        diagram = orchestrator.visualize()

        print("Graph visualization generated")
        print(f"  - Diagram length: {len(diagram)} chars")
        print("\nTo view the graph:")
        print("1. Copy the Mermaid code below")
        print("2. Paste at: https://mermaid.live/")
        print("\n" + "-" * 70)
        print(diagram)
        print("-" * 70)

        return True

    except Exception as e:
        print(f"Visualization error: {e}")
        return False


async def main():
    """Run all tests"""

    print("\n" + "=" * 70)
    print("VETERINARY VOICE-AI SYSTEM TEST SUITE")
    print("=" * 70)
    print("\nThis will test:")
    print("  1. Configuration (API keys)")
    print("  2. RAG System (Pinecone)")
    print("  3. Multi-Agent Orchestrator (LangGraph)")
    print("  4. Graph Visualization")
    print("")

    results = {
        "Configuration": await test_configuration(),
        "RAG System": await test_rag_system(),
        "Orchestrator": await test_orchestrator(),
        "Visualization": await test_graph_visualization()
    }

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {test_name}")

    all_passed = all(results.values())

    if all_passed:
        print("\nALL TESTS PASSED!")
        print("\nSystem is ready to use!")
        print(f"\nStart server with:")
        print(f"  cd backend")
        print(f"  python -m app.main")
    else:
        print("\nSOME TESTS FAILED")
        print("\nPlease fix the issues above before running the system.")

    print("=" * 70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())