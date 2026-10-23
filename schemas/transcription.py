from enum import Enum
from typing import List
from pydantic import BaseModel, Field


class LanguageOption(str, Enum):
    english = "english"
    arabic = "arabic"
    en = "en"
    ar = "ar"


class SegmentDetail(BaseModel):
    start: float = Field(..., description="Start timestamp of the segment in seconds")
    end: float = Field(..., description="End timestamp of the segment in seconds")
    text: str = Field(..., description="Transcribed text within this segment")


class TranscriptionResponse(BaseModel):
    language: str = Field(..., description="Detected or selected language code (e.g. 'en', 'ar')")
    transcript: str = Field(..., description="Full combined transcribed text")
    segments: List[SegmentDetail] = Field(default_factory=list, description="List of timestamped segments")
