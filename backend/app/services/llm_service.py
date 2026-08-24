from groq import Groq

from app.core.exceptions import LLMError

class LLMService:
    """Generate grounded answers with Groq"""

    def __init__(self, client: Groq, model: str, temperature: float, max_tokens: int)-> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate one answer from the language model"""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
        except Exception as error:
            raise LLMError("Failed to generate an answer.") from error

        answer = response.choices[0].message.content

        if not answer:
            raise LLMError("The language model returned an empty answer")

        return answer