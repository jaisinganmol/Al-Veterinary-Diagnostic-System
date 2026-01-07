"""
ElevenLabs Text-to-Speech Service

High-quality voice synthesis for veterinary assistant responses.
"""

import time
from typing import Optional, AsyncIterator
from io import BytesIO

from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

from backend.app.config import settings


class ElevenLabsTTSService:
    """
    ElevenLabs TTS for natural voice output.

    Features:
    - Professional medical voice
    - Low latency generation
    - Streaming support
    """

    def __init__(self):
        print("Initializing ElevenLabs TTS Service...")

        self.client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)

        # Voice configuration
        self.voice_id = settings.ELEVENLABS_VOICE_ID
        self.model = settings.ELEVENLABS_MODEL

        self.voice_settings = VoiceSettings(
            stability=settings.ELEVENLABS_STABILITY,
            similarity_boost=settings.ELEVENLABS_SIMILARITY_BOOST,
            style=0.0,  # Neutral style for medical context
            use_speaker_boost=True
        )

        print(f"ElevenLabs initialized (voice: {self.voice_id})")

    async def synthesize(
            self,
            text: str,
            voice_id: Optional[str] = None
    ) -> bytes:
        """
        Convert text to speech.

        Args:
            text: Text to synthesize
            voice_id: Override default voice

        Returns:
            Audio bytes (MP3 format)
        """

        if voice_id is None:
            voice_id = self.voice_id

        try:
            start_time = time.time()

            # Generate audio
            audio_generator = self.client.generate(
                text=text,
                voice=voice_id,
                model=self.model,
                voice_settings=self.voice_settings
            )

            # Convert generator to bytes
            audio_bytes = b"".join(audio_generator)

            latency_ms = int((time.time() - start_time) * 1000)

            print(f"Generated TTS ({latency_ms}ms, {len(audio_bytes)} bytes)")

            return audio_bytes

        except Exception as e:
            print(f"TTS synthesis error: {e}")
            return b""

    async def synthesize_streaming(
            self,
            text: str,
            voice_id: Optional[str] = None
    ) -> AsyncIterator[bytes]:
        """
        Stream audio generation (for longer responses).

        Args:
            text: Text to synthesize
            voice_id: Override default voice

        Yields:
            Audio chunks
        """

        if voice_id is None:
            voice_id = self.voice_id

        try:
            # Generate audio stream
            audio_stream = self.client.generate(
                text=text,
                voice=voice_id,
                model=self.model,
                voice_settings=self.voice_settings,
                stream=True
            )

            # Yield chunks
            for chunk in audio_stream:
                if chunk:
                    yield chunk

        except Exception as e:
            print(f"TTS streaming error: {e}")
            yield b""


# Singleton instance
_tts_service: Optional[ElevenLabsTTSService] = None


def get_tts_service() -> ElevenLabsTTSService:
    """Get or create TTS service instance"""

    global _tts_service

    if _tts_service is None:
        _tts_service = ElevenLabsTTSService()

    return _tts_service