"""
LLM client for DIAL or local models
"""
import os
from typing import List, Dict, Any
import json

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class LLMClient:
    def __init__(self, use_local: bool = True, model_name: str | None = None):
        """
        Initialize LLM client

        Args:
            use_local: Whether to use local model (True) or DIAL/OpenAI (False)
            model_name: Model name for local or API
        """
        self.use_local = use_local
        self.model_name = model_name or os.getenv(
            "LOCAL_LLM_MODEL",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
        self.client = None
        self.model = None
        self.tokenizer = None
        self.pipeline = None

        if use_local:
            self._setup_local_model()
        else:
            self._setup_api_client()

    def _setup_local_model(self):
        """Set up local Hugging Face model"""
        if not HAS_TRANSFORMERS:
            print("❌ Transformers not installed. Install with: pip install transformers torch")
            return

        try:
            print(f"📥 Loading local model: {self.model_name}")

            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device=-1,
                torch_dtype=torch.float32,
                max_length=512
            )

            print("✅ Local model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load local model: {e}")
            print("⚠️ Falling back to mock responses")

    def _setup_api_client(self):
        """Set up DIAL/OpenAI API client"""
        if not HAS_OPENAI:
            print("❌ OpenAI not installed. Install with: pip install openai")
            return

        try:
            api_key = os.getenv("DIAL_API_KEY") or os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("DIAL_BASE_URL", "https://api.openai.com/v1")

            if not api_key:
                print("⚠️ No API key found. Set DIAL_API_KEY or OPENAI_API_KEY environment variable")
                return

            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            print("✅ API client initialized")
        except Exception as e:
            print(f"❌ Failed to setup API client: {e}")

    def generate_response(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
        """
        Generate response from LLM
        """
        if self.use_local and self.pipeline:
            return self._generate_local(prompt, max_tokens, temperature)
        elif not self.use_local and self.client:
            return self._generate_api(prompt, max_tokens, temperature)
        else:
            return self._generate_mock(prompt)

    def _generate_local(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate response using local model"""
        try:
            result = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.pipeline.tokenizer.eos_token_id,
            )
            return result[0]["generated_text"]
        except Exception as e:
            print(f"❌ Local generation failed: {e}")
            return self._generate_mock(prompt)

    def _generate_api(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate response using API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )

            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ API generation failed: {e}")
            return self._generate_mock(prompt)

    def _generate_mock(self, prompt: str) -> str:
        """Generate mock response for testing"""
        return f"Mock response to: {prompt[:50]}... (Install transformers for local model or set API key for DIAL)"

    def format_rag_prompt(self, question: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Format prompt for RAG
        """
        context_text = "\n\n".join([
            f"Quote: {doc.get('quote', '')}\n"
            f"Author: {doc.get('author', 'Unknown')}\n"
            f"Context: {doc.get('context', '')}\n"
            f"Source: {doc.get('source', '')}"
            for doc in context_docs
        ])

        prompt = f"""You are a historical quotes expert. Answer the user's question based ONLY on the provided context.

Context information:
{context_text}

User Question: {question}

Instructions:
1. Answer based ONLY on the provided context
2. If the answer is not in the context, say "I cannot find that information in my knowledge base"
3. Be concise and accurate
4. Cite sources when possible

Answer:"""

        return prompt


llm_client = LLMClient(use_local=True)

if __name__ == "__main__":
    client = LLMClient(use_local=True)
    test_prompt = "What is the capital of France?"
    response = client.generate_response(test_prompt)
    print(f"Prompt: {test_prompt}")
    print(f"Response: {response}")
