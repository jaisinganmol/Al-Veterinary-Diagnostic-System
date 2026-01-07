"""
LangGraph State Definition

This state object flows through the entire LangGraph workflow,
carrying all data needed for multi-agent orchestration.
"""

from typing import TypedDict, List, Optional, Annotated, Literal, Dict, Any
from operator import add
from datetime import datetime


class VetAgentState(TypedDict):
    """
    Comprehensive state for veterinary AI assistant.

    Performance tracking:
    - RAG confidence for 85% hallucination reduction
    - Latency metrics for sub-2s target
    - Agent execution tracking for 40% workflow improvement
    """

    # ==================== INPUT ====================
    query: str
    """Original voice transcript or text query"""

    audio_file_path: Optional[str]
    """Path to audio file if voice input"""

    user_id: str
    """Unique user identifier"""

    session_id: str
    """Session tracking"""

    timestamp: datetime
    """Query timestamp"""

    # ==================== PATIENT CONTEXT ====================
    species: Optional[str]
    """Species: canine, feline, equine, etc."""

    weight_kg: Optional[float]
    """Patient weight in kilograms"""

    age_years: Optional[float]
    """Patient age in years"""

    breed: Optional[str]
    """Breed information"""

    sex: Optional[str]
    """Male/Female, Intact/Neutered/Spayed"""

    presenting_complaint: Optional[str]
    """Chief complaint"""

    # ==================== ROUTING & CLASSIFICATION ====================
    query_intent: Optional[Literal["triage", "diagnosis", "treatment", "general", "documentation"]]
    """Primary intent of the query"""

    urgency_level: Optional[Literal["emergency", "urgent", "routine"]]
    """Triage urgency classification"""

    requires_multiple_agents: bool
    """Whether query needs multiple specialist agents"""

    routing_confidence: float
    """Confidence in routing decision (0.0-1.0)"""

    # ==================== RAG RETRIEVAL (PINECONE) ====================
    rag_retrieved_docs: Optional[List[str]]
    """Top-k documents from Pinecone"""

    rag_metadata: Optional[List[Dict[str, Any]]]
    """Metadata for retrieved documents"""

    rag_similarity_scores: Optional[List[float]]
    """Cosine similarity scores"""

    rag_confidence_score: float
    """Overall RAG confidence (0.0-1.0) - targets >0.80 for 85% hallucination reduction"""

    rag_namespace_used: Optional[str]
    """Which Pinecone namespace was queried"""

    knowledge_gaps: Annotated[List[str], add]
    """Topics where RAG had insufficient information"""

    # ==================== WEB SEARCH FALLBACK ====================
    requires_web_search: bool
    """True if RAG confidence < threshold"""

    web_search_query: Optional[str]
    """Optimized query for web search"""

    web_search_results: Optional[List[Dict[str, Any]]]
    """Results from Tavily search"""

    web_search_performed: bool
    """Whether web search was actually executed"""

    # ==================== AGENT OUTPUTS ====================
    triage_assessment: Optional[str]
    """Output from Triage Agent"""

    triage_urgency: Optional[str]
    """Urgency determined by triage"""

    diagnosis_differentials: Optional[str]
    """Output from Diagnosis Agent"""

    treatment_plan: Optional[str]
    """Output from Treatment Agent"""

    documentation: Optional[str]
    """SOAP notes or documentation"""

    # ==================== AGENT TRACKING ====================
    agents_invoked: Annotated[List[str], add]
    """List of agents that were called"""

    agent_execution_times: Dict[str, int]
    """Execution time for each agent (ms)"""

    agent_confidence_scores: Dict[str, float]
    """Confidence score from each agent"""

    # ==================== FINAL OUTPUT ====================
    final_response: Optional[str]
    """Synthesized final response"""

    response_confidence: float
    """Overall confidence in response (0.0-1.0)"""

    citations: List[str]
    """Sources cited in response"""

    hallucination_risk_score: float
    """Estimated hallucination risk (0.0-1.0, lower is better)"""

    grounding_quality: Literal["high", "medium", "low"]
    """Quality of knowledge grounding"""

    # ==================== PERFORMANCE METRICS ====================
    total_latency_ms: int
    """Total end-to-end latency (target: <2000ms)"""

    rag_latency_ms: int
    """Time spent on RAG retrieval"""

    agent_orchestration_latency_ms: int
    """Time spent in agent coordination"""

    whisper_transcription_ms: int
    """Whisper STT latency"""

    tts_generation_ms: int
    """TTS generation latency"""

    tokens_consumed: int
    """Total tokens used (for cost tracking)"""

    cache_hit: bool
    """Whether response was served from cache"""

    # ==================== QUALITY METRICS ====================
    diagnostic_accuracy_score: Optional[float]
    """Estimated diagnostic accuracy (if evaluable)"""

    rag_utilization_rate: float
    """% of response grounded in RAG vs. web search"""

    workflow_efficiency_score: float
    """Agent coordination efficiency"""

    # ==================== ERROR HANDLING ====================
    errors: Annotated[List[str], add]
    """Any errors encountered"""

    warnings: Annotated[List[str], add]
    """Non-fatal warnings"""

    retry_count: int
    """Number of retries attempted"""

    error_recovery_performed: bool
    """Whether error recovery was needed"""

    # ==================== DEBUG INFO ====================
    debug_info: Optional[Dict[str, Any]]
    """Additional debug information"""

    graph_path: Optional[List[str]]
    """Path taken through LangGraph nodes"""


# ==================== HELPER FUNCTIONS ====================

def create_initial_state(
        query: str,
        user_id: str = "vet_user",
        session_id: Optional[str] = None,
        species: Optional[str] = None,
        weight_kg: Optional[float] = None,
        age_years: Optional[float] = None,
        breed: Optional[str] = None
) -> VetAgentState:
    """
    Create initial state with defaults.

    Args:
        query: User query
        user_id: User identifier
        session_id: Session ID (generated if None)
        species: Patient species
        weight_kg: Patient weight
        age_years: Patient age
        breed: Patient breed

    Returns:
        Initialized VetAgentState
    """

    if session_id is None:
        session_id = f"session_{datetime.now().timestamp()}"

    return VetAgentState(
        # Input
        query=query,
        audio_file_path=None,
        user_id=user_id,
        session_id=session_id,
        timestamp=datetime.now(),

        # Patient
        species=species,
        weight_kg=weight_kg,
        age_years=age_years,
        breed=breed,
        sex=None,
        presenting_complaint=None,

        # Routing
        query_intent=None,
        urgency_level=None,
        requires_multiple_agents=False,
        routing_confidence=0.0,

        # RAG
        rag_retrieved_docs=None,
        rag_metadata=None,
        rag_similarity_scores=None,
        rag_confidence_score=0.0,
        rag_namespace_used=None,
        knowledge_gaps=[],

        # Web search
        requires_web_search=False,
        web_search_query=None,
        web_search_results=None,
        web_search_performed=False,

        # Agents
        triage_assessment=None,
        triage_urgency=None,
        diagnosis_differentials=None,
        treatment_plan=None,
        documentation=None,

        # Tracking
        agents_invoked=[],
        agent_execution_times={},
        agent_confidence_scores={},

        # Output
        final_response=None,
        response_confidence=0.0,
        citations=[],
        hallucination_risk_score=1.0,
        grounding_quality="low",

        # Performance
        total_latency_ms=0,
        rag_latency_ms=0,
        agent_orchestration_latency_ms=0,
        whisper_transcription_ms=0,
        tts_generation_ms=0,
        tokens_consumed=0,
        cache_hit=False,

        # Quality
        diagnostic_accuracy_score=None,
        rag_utilization_rate=0.0,
        workflow_efficiency_score=0.0,

        # Errors
        errors=[],
        warnings=[],
        retry_count=0,
        error_recovery_performed=False,

        # Debug
        debug_info={},
        graph_path=[]
    )


def calculate_hallucination_risk(state: VetAgentState) -> float:
    """
    Calculate hallucination risk based on RAG grounding.

    Target: 85% reduction through RAG grounding

    Returns:
        Risk score (0.0 = no risk, 1.0 = high risk)
    """

    rag_confidence = state.get("rag_confidence_score", 0.0)
    web_search_performed = state.get("web_search_performed", False)

    # High RAG confidence = low hallucination risk
    if rag_confidence >= 0.85:
        return 0.05  # 5% risk (excellent grounding)
    elif rag_confidence >= 0.75:
        return 0.15  # 15% risk (good grounding)
    elif rag_confidence >= 0.60:
        if web_search_performed:
            return 0.30  # 30% risk (acceptable with web backup)
        else:
            return 0.50  # 50% risk (marginal grounding)
    else:
        if web_search_performed:
            return 0.60  # 60% risk (poor RAG, relying on web)
        else:
            return 0.90  # 90% risk (very poor grounding)


def update_performance_metrics(state: VetAgentState) -> VetAgentState:
    """
    Calculate and update performance metrics.

    Returns:
        Updated state with calculated metrics
    """

    # Calculate hallucination risk
    hallucination_risk = calculate_hallucination_risk(state)

    # Calculate RAG utilization
    rag_conf = state.get("rag_confidence_score", 0.0)
    web_used = 1.0 if state.get("web_search_performed", False) else 0.0
    rag_util = rag_conf / (rag_conf + web_used) if (rag_conf + web_used) > 0 else 0.0

    # Determine grounding quality
    if rag_conf >= 0.80:
        grounding = "high"
    elif rag_conf >= 0.60:
        grounding = "medium"
    else:
        grounding = "low"

    return {
        "hallucination_risk_score": hallucination_risk,
        "rag_utilization_rate": rag_util,
        "grounding_quality": grounding
    }