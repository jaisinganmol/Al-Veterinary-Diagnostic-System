"""
Synthesizer Node - Response Combination

Combines outputs from multiple agents into coherent final response.
Adds citations and calculates quality metrics.
"""

import time
from typing import Dict

from backend.app.graph.state import VetAgentState, calculate_hallucination_risk
from backend.app.rag.pinecone_retriever import get_rag_system


class SynthesizerNode:
    """
    Synthesize final response from agent outputs.

    Responsibilities:
    - Combine triage, diagnosis, treatment responses
    - Add citations
    - Calculate hallucination risk
    - Compute quality metrics
    """

    def __init__(self):
        self.rag_system = get_rag_system()

    async def __call__(self, state: VetAgentState) -> Dict:
        """
        Synthesize final response.

        Returns updated state with:
        - final_response
        - citations
        - hallucination_risk_score
        - response_confidence
        """

        start_time = time.time()

        print(f"\n[Synthesizer] Combining agent responses...")

        # Collect agent outputs
        parts = []

        triage = state.get("triage_assessment")
        if triage:
            parts.append(f"## TRIAGE ASSESSMENT\n{triage}")

        diagnosis = state.get("diagnosis_differentials")
        if diagnosis:
            parts.append(f"\n## DIFFERENTIAL DIAGNOSES\n{diagnosis}")

        treatment = state.get("treatment_plan")
        if treatment:
            parts.append(f"\n## TREATMENT PLAN\n{treatment}")

        # Combine all parts
        final_response = "\n\n".join(parts) if parts else "No response generated."

        # Gather citations
        citations = self._gather_citations(state)

        # Calculate hallucination risk
        rag_docs = state.get("rag_retrieved_docs", [])
        hallucination_risk = self.rag_system.calculate_hallucination_risk(
            response_text=final_response,
            retrieved_docs=rag_docs
        )

        # Calculate response confidence
        rag_conf = state.get("rag_confidence_score", 0.0)
        routing_conf = state.get("routing_confidence", 0.0)
        response_confidence = (rag_conf * 0.7 + routing_conf * 0.3)

        # Determine grounding quality
        if rag_conf >= 0.80:
            grounding = "high"
        elif rag_conf >= 0.60:
            grounding = "medium"
        else:
            grounding = "low"

        latency_ms = int((time.time() - start_time) * 1000)

        # Calculate total latency
        total_latency = (
            state.get("rag_latency_ms", 0) +
            sum(state.get("agent_execution_times", {}).values()) +
            latency_ms
        )

        # Calculate RAG utilization
        web_used = 1.0 if state.get("web_search_performed", False) else 0.0
        rag_util = rag_conf / (rag_conf + web_used) if (rag_conf + web_used) > 0 else 0.0

        print(f"  Agents used: {', '.join(state.get('agents_invoked', []))}")
        print(f"  Hallucination risk: {hallucination_risk:.2%}")
        print(f"  Response confidence: {response_confidence:.2%}")
        print(f"  Total latency: {total_latency}ms")
        print(f"[Synthesizer] Complete ({latency_ms}ms)")

        # Check if we met latency target
        if total_latency > 2000:
            print(f"Latency exceeded target (2000ms): {total_latency}ms")

        return {
            "final_response": final_response,
            "citations": citations,
            "hallucination_risk_score": hallucination_risk,
            "response_confidence": response_confidence,
            "grounding_quality": grounding,
            "total_latency_ms": total_latency,
            "rag_utilization_rate": rag_util,
            "graph_path": state.get("graph_path", []) + ["synthesizer"]
        }

    def _gather_citations(self, state: VetAgentState) -> list:
        """
        Gather all sources cited in responses.

        Returns:
            List of citation strings
        """

        citations = []

        # RAG sources
        if state.get("rag_confidence_score", 0.0) > 0.60:
            citations.append("Veterinary Knowledge Base (RAG)")

        # Web search sources
        if state.get("web_search_performed", False):
            web_results = state.get("web_search_results", [])
            for result in web_results:
                url = result.get("url")
                if url:
                    citations.append(f"Web: {url}")

        # Add metadata sources if available
        metadata = state.get("rag_metadata", [])
        for meta in metadata:
            source = meta.get("source")
            if source and source not in citations:
                citations.append(f"Database: {source}")

        return citations


# Singleton instance
synthesizer_node = SynthesizerNode()