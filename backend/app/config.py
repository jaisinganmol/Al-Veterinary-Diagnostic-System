"""
Configuration for Veterinary Voice-AI System
"""

import os
from typing import Literal
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """
    System configuration with performance targets:
    - 15% higher diagnostic accuracy (RAG grounding)
    - 85% reduction in hallucinations
    - 98% speech recognition accuracy
    - Sub-2s response latency
    """

    # ========== API KEYS ==========
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")

    # ========== MODEL CONFIGURATION ==========
    # Claude Sonnet 4 (best reasoning for medical diagnostics)
    CLAUDE_MODEL: str = "claude-sonnet-4-20241022"
    CLAUDE_TEMPERATURE: float = 0.1
    CLAUDE_MAX_TOKENS: int = 2048

    # Whisper (98% accuracy target)
    WHISPER_MODEL: str = "whisper-1"
    WHISPER_LANGUAGE: str = "en"

    # Embeddings
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536

    # ========== PINECONE CONFIGURATION ==========
    PINECONE_INDEX_NAME: str = "vet-knowledge-base"
    PINECONE_NAMESPACE: str = "veterinary"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"

    # RAG retrieval settings
    RAG_TOP_K: int = 5
    RAG_HIGH_CONFIDENCE_THRESHOLD: float = 0.80  # 85% hallucination reduction
    RAG_LOW_CONFIDENCE_THRESHOLD: float = 0.60

    # ========== ELEVENLABS TTS ==========
    ELEVENLABS_VOICE_ID: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel (professional)
    ELEVENLABS_MODEL: str = "eleven_turbo_v2"
    ELEVENLABS_STABILITY: float = 0.5
    ELEVENLABS_SIMILARITY_BOOST: float = 0.75

    # ========== PERFORMANCE TARGETS ==========
    TARGET_LATENCY_MS: int = 2000  # Sub-2s response
    WHISPER_ACCURACY_TARGET: float = 0.98
    HALLUCINATION_REDUCTION_TARGET: float = 0.85
    DIAGNOSTIC_ACCURACY_IMPROVEMENT: float = 0.15

    # ========== AGENT CONFIGURATION ==========
    # Timeouts for each agent (milliseconds)
    TRIAGE_TIMEOUT_MS: int = 500
    DIAGNOSIS_TIMEOUT_MS: int = 800
    TREATMENT_TIMEOUT_MS: int = 700

    # ========== WEB SEARCH ==========
    TAVILY_MAX_RESULTS: int = 3
    TAVILY_SEARCH_DEPTH: str = "advanced"

    # ========== SERVER CONFIGURATION ==========
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # CORS
    CORS_ORIGINS: list = ["*"]

    # ========== LOGGING ==========
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    # ========== CACHING ==========
    ENABLE_REDIS_CACHE: bool = False
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    CACHE_TTL_SECONDS: int = 3600

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


# Validation on startup
def validate_config():
    """Validate critical configuration"""

    errors = []

    if not settings.ANTHROPIC_API_KEY:
        errors.append("ANTHROPIC_API_KEY not set")

    if not settings.OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY not set")

    if not settings.PINECONE_API_KEY:
        errors.append("PINECONE_API_KEY not set")

    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")

    print("Configuration validated successfully")


# Veterinary-specific prompts
class VetPrompts:
    """
    Curated prompts for veterinary medical context.
    Improves accuracy by providing domain-specific guidance.
    """

    WHISPER_CONTEXT = """
    This is a veterinary medical conversation. Common terms include:
    Cerenia, maropitant, carprofen, meloxicam, gabapentin,
    gastroenteritis, pancreatitis, pyometra, GDV, HGE,
    CBC, chemistry panel, radiographs, ultrasound,
    subcutaneous, intramuscular, intravenous,
    milligrams per kilogram, twice daily, PRN.
    """

    TRIAGE_SYSTEM_PROMPT = """
    You are a veterinary triage specialist with expertise in emergency assessment.

    Your role:
    1. Assess urgency: EMERGENCY / URGENT / ROUTINE
    2. Identify critical red flags
    3. Recommend immediate actions

    EMERGENCY indicators:
    - Difficulty breathing, dyspnea
    - Severe bleeding
    - Known toxin ingestion
    - Seizures, status epilepticus
    - Collapse, unconsciousness
    - Bloat/GDV symptoms
    - Penetrating wounds

    URGENT indicators:
    - Persistent vomiting (>3-4x in 24hrs)
    - Bloody diarrhea
    - Not eating >24-48 hours
    - Significant lethargy with other signs
    - Painful abdomen

    Base ALL assessments on provided RAG knowledge.
    If RAG context is insufficient, clearly state that.
    """

    DIAGNOSIS_SYSTEM_PROMPT = """
    You are a veterinary diagnostic specialist with expertise in differential diagnosis.

    Your role:
    1. Generate ranked differential diagnoses
    2. Provide evidence-based reasoning
    3. Suggest appropriate diagnostic tests

    Guidelines:
    - Base differentials on signalment (species, age, breed)
    - Consider prevalence and likelihood
    - Use RAG knowledge base as primary source
    - Fall back to web search only when RAG insufficient
    - Always provide clinical reasoning

    Response must include:
    1. Top 3-5 differentials (ranked)
    2. Supporting evidence for each
    3. Recommended diagnostics
    4. Source citation (RAG/Web)
    """

    TREATMENT_SYSTEM_PROMPT = """
    You are a veterinary treatment specialist with expertise in pharmacology.

    Your role:
    1. Create evidence-based treatment plans
    2. Calculate precise drug dosing for patient weight
    3. Provide monitoring recommendations

    CRITICAL SAFETY RULES:
    - ALWAYS verify dosing with RAG drug database
    - ALWAYS calculate for specific patient weight
    - ALWAYS include contraindications and warnings
    - If dosing uncertain, state explicitly
    - Never guess or extrapolate dosages

    Response must include:
    1. Immediate interventions
    2. Medications with precise dosing
    3. Route and frequency
    4. Monitoring parameters
    5. Source citation (RAG/FDA label)
    """


vet_prompts = VetPrompts()