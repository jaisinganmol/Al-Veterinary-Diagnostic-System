"""
Application Configuration
frontend/config/settings.py
"""

import os
from typing import List, Dict
import streamlit as st


class Settings:
    """Application settings and configuration"""

    # =============================================================================
    # API CONFIGURATION
    # =============================================================================

    API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")

    # If using Streamlit secrets
    if hasattr(st, 'secrets') and 'API_BASE_URL' in st.secrets:
        API_BASE_URL = st.secrets["API_BASE_URL"]

    # API Endpoints
    HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
    QUERY_ENDPOINT = f"{API_BASE_URL}/api/query"
    TRANSCRIBE_ENDPOINT = f"{API_BASE_URL}/api/voice/transcribe"
    SYNTHESIZE_ENDPOINT = f"{API_BASE_URL}/api/voice/synthesize"
    VOICE_QUERY_ENDPOINT = f"{API_BASE_URL}/api/voice/query"
    GRAPH_VISUALIZE_ENDPOINT = f"{API_BASE_URL}/api/graph/visualize"

    # =============================================================================
    # TIMEOUTS
    # =============================================================================

    HEALTH_CHECK_TIMEOUT = 3
    TRANSCRIPTION_TIMEOUT = 30
    QUERY_TIMEOUT = 60
    TTS_TIMEOUT = 30

    # =============================================================================
    # PERFORMANCE TARGETS
    # =============================================================================

    TARGET_LATENCY_MS = 2000
    TARGET_RAG_CONFIDENCE = 0.80
    TARGET_HALLUCINATION_RISK = 0.15
    TARGET_WHISPER_ACCURACY = 0.98

    HALLUCINATION_REDUCTION_TARGET = 0.85
    DIAGNOSTIC_ACCURACY_IMPROVEMENT = 0.15
    WORKFLOW_IMPROVEMENT = 0.40

    # =============================================================================
    # UI CONFIGURATION
    # =============================================================================

    APP_TITLE = "Veterinary AI Assistant"
    APP_ICON = "🐾"

    # Species options
    SPECIES_OPTIONS: List[str] = [
        "",
        "Canine",
        "Feline",
        "Equine",
        "Avian",
        "Exotic",
        "Bovine",
        "Caprine",
        "Ovine",
        "Porcine"
    ]

    # Weight ranges (kg)
    MIN_WEIGHT = 0.1
    MAX_WEIGHT = 1000.0

    # Age ranges (years)
    MIN_AGE = 0.0
    MAX_AGE = 50.0

    # History settings
    MAX_HISTORY_ITEMS = 20
    MAX_SIDEBAR_HISTORY = 5

    # =============================================================================
    # EXAMPLE QUERIES
    # =============================================================================

    EXAMPLE_QUERIES: Dict[str, str] = {
        "🚨 Emergency: Dog vomiting blood":
            "25kg dog vomiting blood for 6 hours, very lethargic and weak, gums pale, should I go to emergency?",

        "🩺 Diagnosis: Cat not eating":
            "10-year-old domestic shorthair cat hasn't eaten in 3 days, losing weight, hiding under bed, what could be wrong?",

        "💊 Treatment: Cerenia dosing":
            "What is the correct Cerenia dose for a 4.5kg cat with acute vomiting?",

        "⚠️ Medication: Carprofen safety":
            "Is carprofen safe for a 15-year-old Labrador with early kidney disease? Current medications: enalapril.",

        "🐕 General: Puppy vaccinations":
            "What vaccination schedule should I follow for a 10-week-old Golden Retriever puppy?",

        "🔬 Diagnostics: Pancreatitis workup":
            "Dog with acute vomiting, abdominal pain, and anorexia - suspect pancreatitis. What diagnostics should I run?",

        "🏥 Triage: Seizure assessment":
            "3-year-old Border Collie just had first seizure lasting 2 minutes, now appears normal. Emergency or can wait?",

        "💉 Dosing: Gabapentin calculation":
            "Calculate gabapentin dose for pain management in a 32kg German Shepherd with chronic arthritis"
    }

    # =============================================================================
    # THEME COLORS
    # =============================================================================

    PRIMARY_COLOR = "#2E7D32"
    SECONDARY_COLOR = "#66BB6A"
    BACKGROUND_COLOR = "#FFFFFF"
    SECONDARY_BG_COLOR = "#F0F2F6"
    TEXT_COLOR = "#262730"

    # Urgency colors
    EMERGENCY_COLOR = "#d32f2f"
    URGENT_COLOR = "#f57c00"
    ROUTINE_COLOR = "#388e3c"

    # Metric status colors
    GOOD_COLOR = "#4caf50"
    WARNING_COLOR = "#ff9800"
    DANGER_COLOR = "#f44336"

    # =============================================================================
    # MESSAGES
    # =============================================================================

    SYSTEM_OFFLINE_MSG = "⚠️ Backend system is offline. Please ensure the API server is running."
    SYSTEM_ONLINE_MSG = "✅ System operational"

    NO_PATIENT_WARNING = "⚠️ Patient information not provided. Some features may be limited."

    # =============================================================================
    # VALIDATION
    # =============================================================================

    MIN_QUERY_LENGTH = 5
    MAX_QUERY_LENGTH = 1000

    # =============================================================================
    # AUDIO SETTINGS
    # =============================================================================

    AUDIO_SAMPLE_RATE = 16000
    AUDIO_CHANNELS = 1
    AUDIO_FORMAT = "audio/webm"

    # =============================================================================
    # FEATURE FLAGS
    # =============================================================================

    ENABLE_VOICE_INPUT = True
    ENABLE_ANALYTICS = True
    ENABLE_HISTORY = True
    ENABLE_GRAPH_VISUALIZATION = True
    ENABLE_EXPORT = True

    # =============================================================================
    # CACHE SETTINGS
    # =============================================================================

    CACHE_TTL = 300  # 5 minutes

    @classmethod
    def get_api_url(cls, endpoint: str) -> str:
        """Get full API URL for endpoint"""
        return f"{cls.API_BASE_URL}{endpoint}"

    @classmethod
    def is_valid_species(cls, species: str) -> bool:
        """Check if species is valid"""
        return species.lower() in [s.lower() for s in cls.SPECIES_OPTIONS if s]

    @classmethod
    def format_latency_status(cls, latency_ms: int) -> str:
        """Format latency status message"""
        if latency_ms <= cls.TARGET_LATENCY_MS:
            return f"✓ Sub-{cls.TARGET_LATENCY_MS}ms"
        else:
            return f"Target: <{cls.TARGET_LATENCY_MS}ms"

    @classmethod
    def get_urgency_color(cls, urgency: str) -> str:
        """Get color for urgency level"""
        urgency_lower = urgency.lower()
        if urgency_lower == "emergency":
            return cls.EMERGENCY_COLOR
        elif urgency_lower == "urgent":
            return cls.URGENT_COLOR
        else:
            return cls.ROUTINE_COLOR

    @classmethod
    def get_metric_status_color(cls, value: float, threshold: float, invert: bool = False) -> str:
        """
        Get color for metric based on threshold

        Args:
            value: Current value
            threshold: Target threshold
            invert: If True, lower is better (e.g., hallucination risk)
        """
        if invert:
            if value <= threshold:
                return cls.GOOD_COLOR
            elif value <= threshold * 1.5:
                return cls.WARNING_COLOR
            else:
                return cls.DANGER_COLOR
        else:
            if value >= threshold:
                return cls.GOOD_COLOR
            elif value >= threshold * 0.75:
                return cls.WARNING_COLOR
            else:
                return cls.DANGER_COLOR


# Global settings instance
settings = Settings()