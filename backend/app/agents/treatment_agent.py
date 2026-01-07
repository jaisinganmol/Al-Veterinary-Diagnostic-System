"""
Treatment Agent - Treatment Planning Specialist

Creates evidence-based treatment plans with precise drug dosing.
Ensures medication safety through RAG-grounded drug information.
"""

import time
from typing import Dict

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from backend.app.graph.state import VetAgentState
from backend.app.config import settings, vet_prompts


class TreatmentAgent:
    """
    Treatment planning and pharmacology specialist.

    Responsibilities:
    - Create evidence-based treatment plans
    - Calculate precise drug dosing for patient weight
    - Provide contraindications and warnings
    - Suggest monitoring parameters
    """

    def __init__(self):
        self.llm = ChatAnthropic(
            model=settings.CLAUDE_MODEL,
            temperature=0.1,  # Very low for dosing accuracy
            max_tokens=settings.CLAUDE_MAX_TOKENS,
            api_key=settings.ANTHROPIC_API_KEY
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", vet_prompts.TREATMENT_SYSTEM_PROMPT),
            ("human", """Patient Information:
Species: {species}
Breed: {breed}
Weight: {weight_kg} kg (CRITICAL for dosing)
Age: {age_years} years

Query: {query}

Triage Assessment:
{triage_summary}

Working Diagnosis:
{diagnosis_summary}

RAG Drug Database:
{rag_context}

Web Search Results:
{web_context}

Create treatment plan with PRECISE dosing. Respond in this format:

TREATMENT PLAN:

IMMEDIATE (0-2 hours):
1. [Intervention] - [rationale]
2. [Intervention] - [rationale]

MEDICATIONS:

Drug: [Drug Name]
Indication: [Why prescribing]
Dose: [X mg/kg] = [EXACT total dose for this {weight_kg}kg patient]
Route: [SC/PO/IV/IM]
Frequency: [How often, e.g., q12h, q24h]
Duration: [Days/weeks]
Contraindications: [Important warnings]
Side effects to monitor: [What to watch]

[Repeat for each medication]

SHORT-TERM MANAGEMENT (24-72 hours):
- [Action 1]
- [Action 2]

MONITORING:
- [Parameter]: [How often to check]
- [Parameter]: [When to recheck]

RECHECK:
- [When to follow up]

GROUNDING: [RAG Database / FDA Label / Web Search]
CONFIDENCE: [High / Medium / Low]

SAFETY CHECK:
* All doses calculated for patient weight
* Contraindications reviewed
* Drug interactions considered""")
        ])

        self.chain = self.prompt | self.llm

    async def __call__(self, state: VetAgentState) -> Dict:
        """
        Generate treatment plan with precise dosing.

        Returns updated state with:
        - treatment_plan
        """

        # Only run if treatment needed
        if state.get("query_intent") not in ["treatment", "diagnosis", "triage"]:
            print(f"[Treatment] Skipping (intent: {state.get('query_intent')})")
            return {}

        start_time = time.time()

        print(f"\n[Treatment] Creating treatment plan...")

        # Verify weight is available
        weight_kg = state.get("weight_kg")
        if not weight_kg:
            print(f"[Treatment] No patient weight - dosing will be per kg only")

        try:
            # Prepare contexts
            rag_docs = state.get("rag_retrieved_docs", [])
            rag_context = "\n\n".join(rag_docs) if rag_docs else "No drug database"

            web_results = state.get("web_search_results", [])
            web_context = "\n\n".join([
                f"Source: {r.get('url', 'N/A')}\n{r.get('content', '')}"
                for r in web_results
            ]) if web_results else "No web data"

            triage_summary = state.get("triage_assessment", "No triage")
            diagnosis_summary = state.get("diagnosis_differentials", "No diagnosis")

            # Generate treatment plan
            result = await self.chain.ainvoke({
                "species": state.get("species", "unknown"),
                "breed": state.get("breed", "unknown"),
                "weight_kg": weight_kg or "UNKNOWN - provide per-kg dosing only",
                "age_years": state.get("age_years", "unknown"),
                "query": state["query"],
                "triage_summary": triage_summary,
                "diagnosis_summary": diagnosis_summary,
                "rag_context": rag_context,
                "web_context": web_context
            })

            response_text = result.content

            latency_ms = int((time.time() - start_time) * 1000)

            print(f"[Treatment] Complete ({latency_ms}ms)")

            # Update tracking
            execution_times = state.get("agent_execution_times", {})
            execution_times["treatment"] = latency_ms

            return {
                "treatment_plan": response_text,
                "agents_invoked": ["treatment"],
                "agent_execution_times": execution_times,
                "graph_path": state.get("graph_path", []) + ["treatment"]
            }

        except Exception as e:
            print(f"[Treatment] Error: {e}")

            return {
                "treatment_plan": f"Error in treatment planning: {str(e)}",
                "errors": state.get("errors", []) + [f"Treatment error: {str(e)}"],
                "graph_path": state.get("graph_path", []) + ["treatment"]
            }


# Singleton instance
treatment_agent = TreatmentAgent()