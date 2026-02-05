import os
from typing import List, Dict, Any, Optional

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from transformers import pipeline
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class LLMClient:
    def __init__(self, use_local: bool = True, model_name: Optional[str] = None):
        self.use_local = use_local
        self.model_name = model_name or os.getenv(
            "LOCAL_LLM_MODEL",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
        self.client = None
        self.pipeline = None

        if use_local:
            self._setup_local_model()
        else:
            self._setup_api_client()

    def _setup_local_model(self):
        if not HAS_TRANSFORMERS:
            print("❌ Transformers not installed. Install with: pip install transformers torch")
            return

        try:
            print(f"📥 Loading local model: {self.model_name}")

            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device=-1,
                torch_dtype=torch.float32
            )

            print("✅ Local model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load local model: {e}")
            print("⚠️ Falling back to mock responses")

    def _setup_api_client(self):
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

    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.7,
        deterministic: bool = False
    ) -> str:
        if deterministic:
            temperature = 0.0

        if self.use_local and self.pipeline:
            return self._generate_local(prompt, max_tokens, temperature, deterministic)
        elif not self.use_local and self.client:
            return self._generate_api(prompt, max_tokens, temperature)
        else:
            return self._generate_mock(prompt)

    def _generate_local(self, prompt: str, max_tokens: int, temperature: float, deterministic: bool) -> str:
        try:
            result = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=(not deterministic),
                pad_token_id=self.pipeline.tokenizer.eos_token_id
            )
            text = result[0].get("generated_text", "")
            if text.startswith(prompt):
                text = text[len(prompt):]
            return text.strip()
        except Exception as e:
            print(f"❌ Local generation failed: {e}")
            return self._generate_mock(prompt)

    def _generate_api(self, prompt: str, max_tokens: int, temperature: float) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"❌ API generation failed: {e}")
            return self._generate_mock(prompt)

    def _generate_mock(self, prompt: str) -> str:
        return f"Mock response to: {prompt[:50]}... (Install transformers for local model or set API key for DIAL)"

    def format_rag_prompt(self, question: str, context_docs: List[Dict[str, Any]]) -> str:
        context_blocks = []
        for i, doc in enumerate(context_docs):
            label = f"Q{i+1}"
            block = (
                f"{label} - Quote: {doc.get('quote', '')}\n"
                f"Author: {doc.get('author', 'Unknown')}\n"
                f"Context: {doc.get('context', '')}\n"
                f"Source: {doc.get('source', '')}"
            )
            context_blocks.append(block)

        context_text = "\n\n".join(context_blocks)

        prompt = f"""You are a historical quotes expert. Answer the user's question based ONLY on the provided context.

Context information (each quote has an ID like Q1, Q2, etc):
{context_text}

User Question: {question}

Instructions:
1. Answer based ONLY on the provided context
2. If the answer is not in the context, say "I cannot find that information in my knowledge base"
3. When you reference a quote, cite it using its ID in square brackets, e.g. [Q1], [Q2]
4. Be concise and accurate
5. Do NOT invent new quotes or sources that are not in the context

Answer:"""

        return prompt

    def format_rag_prompt_with_analysis(self, question: str, context_docs: List[Dict[str, Any]]) -> str:
        context_entries = []
        for i, doc in enumerate(context_docs):
            entry = [
                f"QUOTE {i+1}:",
                f"  Text: \"{doc.get('quote', '')}\"",
                f"  Author: {doc.get('author', 'Unknown')}",
                f"  Era/Time Period: {doc.get('era', 'Unknown')}",
                f"  Historical Context: {doc.get('context', 'Not specified')}",
                f"  Source/Document: {doc.get('source', 'Not specified')}",
                f"  Topics/Themes: {doc.get('topic', 'Not specified')}",
                f"  Tags/Keywords: {', '.join(doc.get('tags', [])) if doc.get('tags') else 'Not specified'}"
            ]

            interpretation = doc.get('interpretation', '')
            if interpretation:
                entry.append(f"  Interpretation: {interpretation}")

            significance = doc.get('historical_significance', '')
            if significance:
                entry.append(f"  Historical Significance: {significance}")

            themes = doc.get('themes', '')
            if themes:
                entry.append(f"  Key Themes: {themes}")

            context_entries.append("\n".join(entry))

        context_text = "\n\n" + ("-" * 60) + "\n".join(context_entries) + "\n" + ("-" * 60)

        prompt = f"""You are a historian, literary analyst, and educator specializing in historical quotes. 
Your task is to provide COMPREHENSIVE ANALYSIS that includes both factual answers and deeper understanding.

AVAILABLE INFORMATION:{context_text}

USER QUESTION: {question}

ANALYSIS FRAMEWORK - Your response should include these elements:

1. DIRECT ANSWER:
   - Provide clear, factual answer to the question
   - Reference specific quotes and authors
   - Include relevant dates/eras if available

2. QUOTE INTERPRETATION:
   - Explain what each relevant quote MEANS
   - Break down key phrases and their significance
   - Connect to the author's broader philosophy or work

3. HISTORICAL CONTEXT:
   - Describe the historical circumstances when the quote was said
   - Explain relevant events, movements, or conditions
   - Discuss why this quote was significant AT THAT TIME

4. SIGNIFICANCE & LEGACY:
   - Explain why this quote remains important today
   - Discuss its impact on history, culture, or thought
   - Connect to modern relevance or applications

5. THEMATIC ANALYSIS:
   - Identify key themes or ideas in the quote(s)
   - Connect to broader historical or philosophical concepts
   - Compare/contrast with other similar quotes if relevant

GUIDELINES:
- Base ALL analysis on the provided information
- If information is missing, acknowledge the gap but provide best analysis possible
- Use clear, educational language suitable for students and general audience
- Structure your response with clear sections or paragraphs
- Aim for depth and insight, not just surface-level facts
- Do not repeat the context or the instructions in your output

BEGIN YOUR COMPREHENSIVE ANALYSIS:
"""

        return prompt

    def format_simple_prompt(self, question: str, context_docs: List[Dict[str, Any]]) -> str:
        context_text = "\n\n".join([
            f"\"{doc.get('quote', '')}\" - {doc.get('author', 'Unknown')}"
            for doc in context_docs
        ])

        return f"""Based on these quotes: {context_text}

Question: {question}
Answer: """

    def analyze_quote_meaning(self, quote: str, author: str, context: str = "") -> str:
        prompt = f"""Analyze this historical quote:

QUOTE: "{quote}"
AUTHOR: {author}
CONTEXT: {context if context else "Not provided"}

Please provide:
1. Literal meaning and key phrases
2. Historical significance 
3. Why it was important when said
4. Modern relevance

Analysis:"""

        return self.generate_response(prompt, max_tokens=400, temperature=0.3)

    def compare_quotes(self, quote1: Dict, quote2: Dict) -> str:
        prompt = f"""Compare these two historical quotes:

QUOTE 1: "{quote1.get('quote', '')}"
AUTHOR: {quote1.get('author', 'Unknown')}
CONTEXT: {quote1.get('context', 'Not provided')}

QUOTE 2: "{quote2.get('quote', '')}"
AUTHOR: {quote2.get('author', 'Unknown')}
CONTEXT: {quote2.get('context', 'Not provided')}

Analyze:
1. Similarities in themes or messages
2. Differences in historical context
3. Different approaches to similar ideas
4. Relative historical significance

Comparative Analysis:"""

        return self.generate_response(prompt, max_tokens=500, temperature=0.4)


llm_client = LLMClient(use_local=True)


if __name__ == "__main__":
    client = LLMClient(use_local=True)
    test_question = "What did Roosevelt say about fear?"
    test_docs = [
        {
            "quote": "The only thing we have to fear is fear itself.",
            "author": "Franklin D. Roosevelt",
            "context": "First inaugural address during the Great Depression",
            "source": "March 4, 1933",
            "era": "1933",
            "topic": "Leadership, Courage",
            "tags": ["courage", "depression", "leadership"],
            "interpretation": "Encourages facing challenges with courage rather than being paralyzed by anxiety",
            "historical_significance": "Defining speech of New Deal era, inspired nation during economic crisis"
        }
    ]

    basic_prompt = client.format_rag_prompt(test_question, test_docs)
    enhanced_prompt = client.format_rag_prompt_with_analysis(test_question, test_docs)

    basic = client.generate_response(basic_prompt, max_tokens=120, deterministic=True)
    enhanced = client.generate_response(enhanced_prompt, max_tokens=200, deterministic=True)

    print("BASIC:", basic[:300])
    print("ENHANCED:", enhanced[:300])
