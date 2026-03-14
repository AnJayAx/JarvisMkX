"""
RAG Pipeline Module for Jarvis Mk.X (v4)
Fixes: reasoning beyond literal text, application questions, conversational handling.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from processor import PaperProcessor, ProcessedPaper
from retriever import Retriever, RetrievalResult
from typing import List, Optional, Dict
from dataclasses import dataclass, field
import time


# ─── System Prompt (v4 — allows reasoning and inference) ───

SYSTEM_PROMPT = (
    "You are Jarvis Mk.X, a research paper Q&A assistant. "
    "You answer questions based on the provided context from a research paper. "
    "Always explain concepts in your own words rather than copying text directly. "
    "Synthesize information from the context into clear, natural explanations. "
    "When a user asks you to apply the paper's findings to a new situation, "
    "use the paper's ideas and reasoning to provide a thoughtful answer, "
    "clearly noting which parts come from the paper and which are your inference. "
    "If the context truly contains no relevant information at all, say so honestly. "
    "Reference which section or page the information comes from when relevant."
)


@dataclass
class BotResponse:
    answer: str
    sources: List[RetrievalResult]
    context_used: str
    paper_title: str
    confidence: float = 0.0
    generation_time: float = 0.0


@dataclass
class ConversationTurn:
    role: str
    content: str


class JarvisBot:

    def __init__(
        self,
        base_model_name="mistralai/Mistral-7B-Instruct-v0.3",
        adapter_path="models/jarvis-mkx-adapter-v2",
        embed_model_name="sentence-transformers/allenai-specter",
        chunk_size=512,
        chunk_overlap=50,
        load_in_4bit=True,
    ):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.load_in_4bit = load_in_4bit
        self.processor = PaperProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.retriever = Retriever(model_name=embed_model_name)
        self.model = None
        self.tokenizer = None
        self.current_paper = None
        self.conversation_history = []
        self.max_history_turns = 6
        self.corrections = []

    def load_model(self):
        if self.model is not None:
            print("Model already loaded.")
            return
        print(f"Loading base model: {self.base_model_name}")
        bnb_config = None
        if self.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        model_kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
        base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name, **model_kwargs)
        print(f"Loading LoRA adapter: {self.adapter_path}")
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        self.model.eval()
        print("Model loaded and ready.")

    def load_paper(self, pdf_path):
        print(f"Processing PDF: {pdf_path}")
        self.current_paper = self.processor.process(pdf_path)
        print(f"Building search index for {len(self.current_paper.chunks)} chunks...")
        self.retriever.build_index(self.current_paper.chunks)
        self.conversation_history = []
        self.corrections = []
        paper_info = {
            "title": self.current_paper.title,
            "authors": self.current_paper.authors,
            "abstract": self.current_paper.abstract[:300],
            "num_pages": self.current_paper.num_pages,
            "num_sections": len(self.current_paper.sections),
            "num_chunks": len(self.current_paper.chunks),
            "sections": list(self.current_paper.sections.keys()),
        }
        print(f"Paper loaded: {paper_info['title']}")
        print(f"  {paper_info['num_sections']} sections, {paper_info['num_chunks']} chunks")
        return paper_info

    # ─── Question Classification ───

    def _classify_question(self, question):
        """
        Classify the question type to determine handling strategy.
        Returns: 'meta', 'application', 'conversational', or 'factual'
        """
        q_lower = question.lower().strip()

        # Meta questions (summarize, overview, etc.)
        meta_patterns = [
            "summarize", "summary", "what is this paper about",
            "what is the paper about", "overview", "main topic",
            "what does this paper discuss", "key points", "key takeaways",
            "tell me about this paper", "what is the research about",
            "abstract", "main idea", "main contribution", "main finding",
            "what is the main", "brief description", "describe this paper",
            "what are the most important", "most important takeaways",
            "optimistic or pessimistic", "overall conclusion",
            "one key change", "key recommendation", "main recommendation",
            "what would you recommend", "overall message",
            "what can we learn", "what did you learn",
            "give me a summary", "paper summary", "research summary",
            "what are the key", "what is the conclusion",
            "do you think this paper", "what do you think",
        ]
        if any(pattern in q_lower for pattern in meta_patterns):
            return "meta"

        # Application questions (apply paper's ideas to new scenario)
        application_patterns = [
            "what would the paper suggest", "what would the author suggest",
            "what solutions", "how could", "how would",
            "what would you recommend", "based on this paper",
            "according to the paper, how", "given that",
            "what can be done", "what should", "if we apply",
            "for a country like", "for a company like",
            "how can we use", "what implications",
            "how does this apply", "what are the solutions",
            "what advice", "suggest", "recommend",
        ]
        if any(pattern in q_lower for pattern in application_patterns):
            return "application"

        # Conversational (greetings, insults, off-topic chitchat)
        conversational_patterns = [
            "hello", "hi ", "hey", "how are you", "thank",
            "you are not", "you're not", "you suck", "stupid",
            "who are you", "what are you", "goodbye", "bye",
            "good morning", "good evening",
        ]
        if any(pattern in q_lower for pattern in conversational_patterns):
            return "conversational"

        return "factual"

    # ─── Context Builders ───

    def _get_meta_context(self):
        """Build context from abstract + introduction + conclusion."""
        parts = []
        for section_name, content in self.current_paper.sections.items():
            name_lower = section_name.lower()
            if any(kw in name_lower for kw in [
                "abstract", "introduction", "conclusion", "preamble", "summary"
            ]):
                text = content[:2000] if len(content) > 2000 else content
                parts.append(f"[Section: {section_name}]\n{text}")
        if parts:
            return "\n\n---\n\n".join(parts)
        fallback = []
        if self.current_paper.abstract:
            fallback.append(f"[Abstract]\n{self.current_paper.abstract}")
        for chunk in self.current_paper.chunks[:3]:
            fallback.append(f"[{chunk.section}]\n{chunk.text[:1000]}")
        return "\n\n---\n\n".join(fallback)

    def _get_broad_context(self, question, max_tokens=2000):
        """
        For application questions, cast a wider net:
        retrieve more chunks AND include conclusion/recommendations.
        """
        # Get retrieval results with relaxed threshold
        sources = self.retriever.retrieve(question, top_k=8)

        # Always include conclusion/recommendation sections
        priority_sections = []
        for section_name, content in self.current_paper.sections.items():
            name_lower = section_name.lower()
            if any(kw in name_lower for kw in [
                "conclusion", "recommendation", "discussion", "future",
                "implication", "summary"
            ]):
                text = content[:1500] if len(content) > 1500 else content
                priority_sections.append(f"[Section: {section_name}]\n{text}")

        # Also include top retrieval results
        retrieval_parts = []
        total_tokens = 0
        for result in sources[:5]:
            approx_tokens = len(result.text) // 4
            if total_tokens + approx_tokens > max_tokens // 2:
                break
            source_info = f"[Section: {result.section} | Page(s): {', '.join(map(str, result.page_numbers))}]"
            retrieval_parts.append(f"{source_info}\n{result.text}")
            total_tokens += approx_tokens

        # Combine: priority sections first, then retrieval results
        all_parts = priority_sections + retrieval_parts

        # Deduplicate (rough check)
        seen = set()
        unique_parts = []
        for part in all_parts:
            key = part[:100]
            if key not in seen:
                seen.add(key)
                unique_parts.append(part)

        return "\n\n---\n\n".join(unique_parts), sources

    # ─── Core Generation ───

    def _generate_answer(self, question, context, corrections_text="",
                         max_new_tokens=512, temperature=0.3):
        messages = self._build_messages(question, context, corrections_text)
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
            return_dict=False,
        )
        # Ensure inputs is a tensor, not a BatchEncoding
        if not isinstance(inputs, torch.Tensor):
            inputs = inputs["input_ids"]
        inputs = inputs.to(self.model.device)

        if inputs.shape[-1] > 2048 - max_new_tokens:
            self.conversation_history = self.conversation_history[-2:]
            messages = self._build_messages(question, context, corrections_text)
            inputs = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
                return_dict=False,
            )
            if not isinstance(inputs, torch.Tensor):
                inputs = inputs["input_ids"]
            inputs = inputs.to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs, max_new_tokens=max_new_tokens,
                temperature=max(temperature, 0.01),
                top_p=0.9, do_sample=True, repetition_penalty=1.1,
            )
        return self.tokenizer.decode(
            outputs[0][inputs.shape[-1]:], skip_special_tokens=True
        ).strip()

    # ─── Main Ask Method ───

    def ask(self, question, top_k=5, max_context_tokens=1500,
            max_new_tokens=512, leniency=50):
        if self.model is None:
            self.load_model()
        if self.current_paper is None:
            return BotResponse(
                answer="No paper loaded. Please upload a PDF first.",
                sources=[], context_used="", paper_title="",
            )

        start_time = time.time()

        # Build corrections text
        corrections_text = ""
        if self.corrections:
            relevant = self._find_relevant_corrections(question)
            if relevant:
                corrections_text = (
                    "\n\n### Important Corrections (from user feedback):\n"
                    + "\n".join(
                        f"- Q: {c['question']} -> Correct answer: {c['correction']}"
                        for c in relevant
                    )
                )

        # ─── Classify the question ───
        q_type = self._classify_question(question)

        # ─── CONVERSATIONAL: handle greetings, off-topic ───
        if q_type == "conversational":
            q_lower = question.lower()
            if any(w in q_lower for w in ["hello", "hi ", "hey", "good morning", "good evening"]):
                answer = (
                    f"Hello! I'm Jarvis Mk.X, your research paper assistant. "
                    f"I currently have \"{self.current_paper.title}\" loaded. "
                    f"Ask me anything about this paper!"
                )
            elif any(w in q_lower for w in ["thank"]):
                answer = "You're welcome! Let me know if you have more questions about the paper."
            elif any(w in q_lower for w in ["who are you", "what are you"]):
                answer = (
                    "I'm Jarvis Mk.X, a research paper Q&A assistant. "
                    "I can read and analyze research papers, answer questions about their content, "
                    "and help you understand their findings. Upload a PDF and ask away!"
                )
            elif any(w in q_lower for w in ["bye", "goodbye"]):
                answer = "Goodbye! Feel free to come back anytime."
            else:
                answer = (
                    "I'm designed to answer questions about research papers. "
                    f"I currently have \"{self.current_paper.title}\" loaded. "
                    "Try asking me about its findings, methodology, or conclusions!"
                )
            self._add_to_history(question, answer)
            return BotResponse(
                answer=answer, sources=[], context_used="",
                paper_title=self.current_paper.title,
                confidence=1.0, generation_time=time.time() - start_time,
            )

        # ─── META: summary, overview, key takeaways ───
        if q_type == "meta":
            context = self._get_meta_context()
            sources = self.retriever.retrieve(question, top_k=3)
            answer = self._generate_answer(
                question, context, corrections_text,
                max_new_tokens=max_new_tokens, temperature=0.3,
            )
            self._add_to_history(question, answer)
            return BotResponse(
                answer=answer, sources=sources, context_used=context,
                paper_title=self.current_paper.title,
                confidence=1.0, generation_time=time.time() - start_time,
            )

        # ─── APPLICATION: apply paper's ideas to new scenario ───
        if q_type == "application":
            context, sources = self._get_broad_context(question)

            # Add a reasoning instruction to the context
            reasoning_note = (
                "\n\n### Note: The user is asking you to apply the paper's findings "
                "to a specific scenario. Use the paper's ideas, recommendations, and "
                "conclusions to provide a thoughtful answer. Clearly state which parts "
                "come from the paper and which are your reasoning."
            )
            context = context + reasoning_note

            answer = self._generate_answer(
                question, context, corrections_text,
                max_new_tokens=max_new_tokens, temperature=0.35,
            )
            self._add_to_history(question, answer)

            filtered = [s for s in sources if s.score >= 0.1]
            avg_conf = sum(s.score for s in filtered) / len(filtered) if filtered else 0.0

            return BotResponse(
                answer=answer, sources=sources, context_used=context,
                paper_title=self.current_paper.title,
                confidence=avg_conf, generation_time=time.time() - start_time,
            )

        # ─── FACTUAL: standard retrieval-based Q&A ───
        score_threshold = 0.5 - (leniency / 100) * 0.45
        temperature = 0.1 + (leniency / 100) * 0.4

        sources = self.retriever.retrieve(question, top_k=top_k)
        filtered_sources = [s for s in sources if s.score >= score_threshold]

        avg_confidence = (
            sum(s.score for s in filtered_sources) / len(filtered_sources)
            if filtered_sources else 0.0
        )

        # If retrieval confidence is low, still attempt an answer using the best available context.
        # This prevents the UI from feeling "broken" for harder questions or weaker embedding models.
        if not filtered_sources:
            if not sources:
                return BotResponse(
                    answer=(
                        "I couldn't retrieve any relevant passages to answer this question. "
                        "Try rephrasing, uploading a clearer PDF, or asking a more specific question."
                    ),
                    sources=[],
                    context_used="",
                    paper_title=self.current_paper.title,
                    confidence=0.0,
                    generation_time=time.time() - start_time,
                )

            best_score = float(sources[0].score)
            fallback_sources = sources[: min(3, len(sources))]

            context_parts = []
            total_tokens = 0
            for result in fallback_sources:
                approx_tokens = len(result.text) // 4
                if total_tokens + approx_tokens > max_context_tokens:
                    break
                source_info = f"[Section: {result.section} | Page(s): {', '.join(map(str, result.page_numbers))}]"
                context_parts.append(f"{source_info}\n{result.text}")
                total_tokens += approx_tokens
            context = "\n\n---\n\n".join(context_parts)

            answer = self._generate_answer(
                question,
                context,
                corrections_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            answer = (
                "⚠️ Low retrieval confidence: I may be missing supporting passages in the PDF. "
                "Answering using the closest matches I found.\n\n" + answer
            )

            self._add_to_history(question, answer)

            return BotResponse(
                answer=answer,
                sources=fallback_sources,
                context_used=context,
                paper_title=self.current_paper.title,
                confidence=best_score,
                generation_time=time.time() - start_time,
            )

        # Build context
        context_parts = []
        total_tokens = 0
        for result in filtered_sources:
            approx_tokens = len(result.text) // 4
            if total_tokens + approx_tokens > max_context_tokens:
                break
            source_info = f"[Section: {result.section} | Page(s): {', '.join(map(str, result.page_numbers))}]"
            context_parts.append(f"{source_info}\n{result.text}")
            total_tokens += approx_tokens
        context = "\n\n---\n\n".join(context_parts)

        answer = self._generate_answer(
            question, context, corrections_text,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )

        self._add_to_history(question, answer)

        return BotResponse(
            answer=answer, sources=filtered_sources, context_used=context,
            paper_title=self.current_paper.title,
            confidence=avg_confidence, generation_time=time.time() - start_time,
        )

    # ─── Conversation Memory ───

    def _build_messages(self, question, context, corrections_text=""):
        messages = []
        first_user_content = (
            f"{SYSTEM_PROMPT}\n\n"
            f"### Context from Research Paper:\n{context}"
            f"{corrections_text}\n\n"
        )
        if self.conversation_history:
            first_user_content += f"### Question:\n{self.conversation_history[0].content}"
            messages.append({"role": "user", "content": first_user_content})
            for turn in self.conversation_history[1:]:
                messages.append({"role": turn.role, "content": turn.content})
            messages.append({"role": "user", "content": question})
        else:
            first_user_content += f"### Question:\n{question}"
            messages.append({"role": "user", "content": first_user_content})
        return messages

    def _add_to_history(self, question, answer):
        self.conversation_history.append(ConversationTurn(role="user", content=question))
        self.conversation_history.append(ConversationTurn(role="assistant", content=answer))
        if len(self.conversation_history) > self.max_history_turns:
            self.conversation_history = self.conversation_history[-self.max_history_turns:]

    def clear_history(self):
        self.conversation_history = []
        print("Conversation history cleared.")

    def get_history(self):
        return [{"role": t.role, "content": t.content} for t in self.conversation_history]

    # ─── Answer Correction ───

    def correct_answer(self, correction):
        if len(self.conversation_history) < 2:
            return "No previous question to correct."
        last_question = self.conversation_history[-2].content
        last_answer = self.conversation_history[-1].content
        self.corrections.append({
            "question": last_question,
            "wrong_answer": last_answer,
            "correction": correction,
        })
        self.conversation_history[-1] = ConversationTurn(
            role="assistant", content=f"{correction} (corrected by user)"
        )
        return (
            f"Thank you for the correction. I've noted that for "
            f"'{last_question[:50]}...', the correct information is: {correction}. "
            f"I'll use this in future answers during this session."
        )

    def _find_relevant_corrections(self, question):
        if not self.corrections:
            return []
        question_words = set(question.lower().split())
        return [
            c for c in self.corrections
            if len(question_words & set(c['question'].lower().split())) >= 2
        ]

    def get_corrections(self):
        return self.corrections

    # ─── Utility ───

    def get_paper_info(self):
        if self.current_paper is None:
            return None
        return {
            "title": self.current_paper.title,
            "authors": self.current_paper.authors,
            "abstract": self.current_paper.abstract,
            "num_pages": self.current_paper.num_pages,
            "sections": list(self.current_paper.sections.keys()),
            "num_chunks": len(self.current_paper.chunks),
        }

    def is_paper_loaded(self):
        return self.current_paper is not None

    def is_model_loaded(self):
        return self.model is not None

    def get_leniency_info(self, leniency):
        score_threshold = 0.5 - (leniency / 100) * 0.45
        temperature = 0.1 + (leniency / 100) * 0.4
        return {
            "leniency": leniency,
            "score_threshold": round(score_threshold, 3),
            "temperature": round(temperature, 2),
            "description": (
                "Very strict - only answers with high confidence"
                if leniency < 20 else
                "Strict - prefers confident answers"
                if leniency < 40 else
                "Balanced - good mix of confidence and coverage"
                if leniency < 60 else
                "Lenient - attempts most questions"
                if leniency < 80 else
                "Very lenient - always attempts an answer"
            ),
        }
