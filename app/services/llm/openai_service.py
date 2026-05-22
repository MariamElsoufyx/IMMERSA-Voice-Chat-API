from openai import OpenAI
import app.core.config as config

class LLMOpenAIService:
    def __init__(self, client=None):
        self.client = client
        self.model_name = config.openAI_model_name
        self.max_completion_tokens = config.openAI_max_completion_tokens

    def generate_reply(self, user_prompt, system_prompt, history=None):
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_completion_tokens=self.max_completion_tokens,
                response_format={"type": "json_object"},
                stream=True,
            )
            tokens = []
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    tokens.append(delta)
            return "".join(tokens).strip()
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            return ""
