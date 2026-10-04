"""
Grammar Correction module (from Sign-Bridge contribution)
Transforms raw sign word sequences into natural English sentences.
Includes lightweight local rule-based corrector and optional HuggingFace T5 transformer model.
"""
import re

class GrammarCorrector:
    def __init__(self, use_llm=False):
        self.use_llm = use_llm
        self.llm_pipeline = None

        if use_llm:
            try:
                from transformers import pipeline
                self.llm_pipeline = pipeline("text2text-generation", model="psg/grammar-correction-t5-small")
            except Exception as e:
                print(f"[GrammarCorrector] Fallback to rule-based engine: {e}")
                self.use_llm = False

    def correct_sentence(self, text: str) -> str:
        """
        Takes raw constructed text and applies grammar correction.
        """
        if not text or not text.strip():
            return ""

        text = text.strip()

        # If LLM model is available, use it
        if self.use_llm and self.llm_pipeline is not None:
            try:
                prompt = f"grammar: {text}"
                res = self.llm_pipeline(prompt, max_length=128)
                return res[0]['generated_text'].strip()
            except Exception as e:
                print(f"[GrammarCorrector] LLM inference failed, fallback to rules: {e}")

        # Lightweight rule-based correction for student laptops
        return self._rule_based_correction(text)

    def _rule_based_correction(self, text: str) -> str:
        words = text.split()
        if not words:
            return ""

        # Common ASL sign replacements to English
        replacements = {
            "i": "I",
            "me": "I",
            "go market": "went to the market",
            "go school": "went to school",
            "drink water": "drank water",
            "need help": "need help",
            "thank": "thank you"
        }

        sentence = " ".join(words)

        # Apply phrase replacements
        for key, val in replacements.items():
            if sentence.lower() == key:
                sentence = val
                break

        # Capitalize first letter
        sentence = sentence[0].upper() + sentence[1:]

        # Add period if missing end punctuation
        if not re.search(r'[.!?]$', sentence):
            sentence += "."

        return sentence
