import os
import sys

class BaseAgent:
    """
    Abstract base class for all clinical agents.
    Exposes unified logging and LLM API client interfaces with automatic fallback.
    """
    def __init__(self, name):
        self.name = name

    def log_action(self, message):
        """
        Standardized console log for agent actions.
        """
        print(f"[{self.name}] {message}")

    def query_llm(self, prompt, system_instruction=None):
        """
        Queries an LLM (Gemini or OpenAI) if environment variables are configured.
        Returns the generated text response, or None if no keys are found or a query fails.
        """
        gemini_key = os.environ.get("GEMINI_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        
        # 1. Try Gemini API
        if gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                
                # Setup configuration
                generation_config = {
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "max_output_tokens": 512,
                }
                
                model = genai.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    generation_config=generation_config,
                    system_instruction=system_instruction
                )
                
                response = model.generate_content(prompt)
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                self.log_action(f"WARNING: Gemini LLM call failed: {e}. Checking other keys...")
                
        # 2. Try OpenAI API
        if openai_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=openai_key)
                
                messages = []
                if system_instruction:
                    messages.append({"role": "system", "content": system_instruction})
                messages.append({"role": "user", "content": prompt})
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.2,
                    max_tokens=512
                )
                
                if response and response.choices:
                    return response.choices[0].message.content.strip()
            except Exception as e:
                self.log_action(f"WARNING: OpenAI LLM call failed: {e}.")

        # 3. If no LLM keys are configured or API calls failed
        return None
