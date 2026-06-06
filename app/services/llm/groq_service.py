from groq import Groq
import app.core.config as config

class LLMGroqService:
    def __init__(self, client=None):
        self.client = client
        self.model_name = config.groq_model_name
        self.max_completion_tokens = config.groq_Max_completion_tokens
        print(f"✅ [LLM] Groq LLM ready (model={self.model_name})")

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
                response_format={"type": "json_object"}, #3lashan mykharafsh w ytla3 shakl tany 
                stream=True,
            )
            tokens = []
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    tokens.append(delta)
            return "".join(tokens).strip().replace("\n", " ")
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            return ""
