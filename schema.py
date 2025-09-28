from pydantic import BaseModel, Field
from typing import List


class Reflection(BaseModel):
    missing: str = Field(description="Critique on missing information")
    superfluous: str = Field(description="Critique on superfluous information")


class AnswerWithCritique(BaseModel):
    answer: str = Field(description="~250 words answer to the question")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching important missing information to address the critiques of current answer"
    )
    reflection: Reflection = Field(description="Reflection on the initial answer")


class ReviseAnswer(AnswerWithCritique):
    """Revise your original answer to your question"""

    references: List[str] = Field(
        description="Citations motivating your updated answer"
    )
