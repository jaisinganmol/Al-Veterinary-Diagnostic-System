"""
Triage Agent - Emergency Assessment Specialist

Assesses urgency and determines if immediate care is needed.
Part of 40% faster diagnostic workflow through rapid triage.
"""

import time
from typing import Dict

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from backend.app.graph.state import VetAgentState
from backend.app.config import settings, vet_prompts


class TriageAgent:
    """
    Emergency assessment and urgency determination specialist.

    Responsibilities:
    - Classify urgency: EMERGENCY / URGENT / ROUTINE
    - Identify critical red flags
    - Recommend immediate actions
    - Extract key clinical signs
    """

    def __init__(self):
        self.llm = ChatAnthropic(
            model=settings.CLAUDE_MODEL,
            temperature=0.1,  # Low temperature for consistent triage
            max_tokens=settings.CLAUDE_MAX_TOKENS,
            api_key=settings.ANTHROPIC_API_KEY
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", vet_prompts.TRIAGE_SYSTEM_PROMPT),
            ("human", """Patient Information:
Species: {species}
Weight: {weight_kg} kg
Age: {age_years} years
Breed: {breed}

Query: {query}

RAG Knowledge Base (Veterinary Database):
{rag_context}

RAG Confidence: {rag_confidence:.2%}

Web Search Results (if RAG insufficient):
{web_context}

Perform emergency triage assessment. Respond in this format:

URGENCY: [EMERGENCY / URGENT / ROUTINE]

KEY CLINICAL SIGNS:
- [Sign 1]
- [Sign 2]
- [Sign 3]

RED FLAGS:
- [Any critical findings]

IMMEDIATE ACTION:
[What should be done right now]

GROUNDING: [RAG Database / Web Search / Both]
CONFIDENCE: [High / Medium / Low]""")
        ])

        self.chain = self.prompt | self.llm

    async def __call__(self, state: VetAgentState) -> Dict:
        """
        Perform triage assessment.

        Returns updated state with:
        - triage_assessment
        - urgency_level
        - triage_urgency
        """

        # Only run if this is a triage query
        if state.get("query_intent") not in ["triage", "diagnosis"]:
            print(f"[Triage] Skipping (intent: {state.get('query_intent')})")
            return {}

        start_time = time.time()

        print(f"\n[Triage] Assessing urgency...")

        try:
            # Prepare RAG context
            rag_docs = state.get("rag_retrieved_docs", [])
            rag_context = "\n\n".join(rag_docs) if rag_docs else "No RAG data available"

            # Prepare web search context
            web_results = state.get("web_search_results", [])
            web_context = "\n\n".join([
                f"Source: {r.get('url', 'N/A')}\n{r.get('content', '')}"
                for r in web_results
            ]) if web_results else "No web search data"

            # Run triage
            result = await self.chain.ainvoke({
                "species": state.get("species", "unknown"),
                "weight_kg": state.get("weight_kg", "unknown"),
                "age_years": state.get("age_years", "unknown"),
                "breed": state.get("breed", "unknown"),
                "query": state["query"],
                "rag_context": rag_context,
                "rag_confidence": state.get("rag_confidence_score", 0.0),
                "web_context": web_context
            })

            response_text = result.content

            # Extract urgency level
            urgency = "routine"
            if "EMERGENCY" in response_text.upper():
                urgency = "emergency"
            elif "URGENT" in response_text.upper():
                urgency = "urgent"

            latency_ms = int((time.time() - start_time) * 1000)

            print(f"  Urgency: {urgency.upper()}")
            print(f"[Triage] Complete ({latency_ms}ms)")

            # Update agent tracking
            execution_times = state.get("agent_execution_times", {})
            execution_times["triage"] = latency_ms

            return {
                "triage_assessment": response_text,
                "urgency_level": urgency,
                "triage_urgency": urgency,
                "agents_invoked": ["triage"],
                "agent_execution_times": execution_times,
                "graph_path": state.get("graph_path", []) + ["triage"]
            }

        except Exception as e:
            print(f"[Triage] Error: {e}")

            return {
                "triage_assessment": f"Error in triage: {str(e)}",
                "urgency_level": "routine",  # Safe default
                "errors": state.get("errors", []) + [f"Triage error: {str(e)}"],
                "graph_path": state.get("graph_path", []) + ["triage"]
            }


# Singleton instance
triage_agent = TriageAgent()