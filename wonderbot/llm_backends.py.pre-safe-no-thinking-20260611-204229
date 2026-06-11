from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import re
import time
from typing import List, Protocol, Sequence

from .config import BackendConfig
from .event_codec import EventCodec
from .hardware import choose_component_device, ensure_offload_dir, parse_torch_dtype
from .memory import MemoryItem


@dataclass(slots=True)
class BackendResult:
    text: str
    backend_name: str
    metadata: dict[str, object] = field(default_factory=dict)


class TextBackend(Protocol):
    def generate(self, stimulus: str, memories: List[MemoryItem], style: str, spontaneous: bool = False) -> BackendResult:
        ...


@dataclass(slots=True)
class ImaginationProposal:
    accepted: bool
    anchor_text: str
    drift: float
    novelty: float
    repetition: float
    hint: str
    associated_texts: List[str]


class LVTCPlanner:
    def __init__(
        self,
        codec: EventCodec,
        delta_scale: float = 0.24,
        creative_depth: int = 1,
        anchor_pullback: float = 0.72,
        novelty_threshold: float = 0.10,
        drift_threshold: float = 0.58,
        repetition_threshold: float = 0.34,
        latency_budget_ms: int = 30,
    ) -> None:
        self.codec = codec
        self.delta_scale = delta_scale
        self.creative_depth = max(1, creative_depth)
        self.anchor_pullback = anchor_pullback
        self.novelty_threshold = novelty_threshold
        self.drift_threshold = drift_threshold
        self.repetition_threshold = repetition_threshold
        self.latency_budget_ms = latency_budget_ms

    def propose(self, stimulus: str, memories: List[MemoryItem], spontaneous: bool = False) -> ImaginationProposal:
        started = time.perf_counter_ns()
        anchor_text = stimulus.strip() or (memories[0].text if memories else "")
        if not anchor_text or not memories:
            return ImaginationProposal(False, anchor_text, 0.0, 0.0, 0.0, "", [])

        anchor_vector = self.codec.vectorize(anchor_text)
        candidates = self._select_associative_memories(anchor_vector, memories)
        if not candidates:
            return ImaginationProposal(False, anchor_text, 0.0, 0.0, 0.0, "", [])

        assoc_vector = _normalize_vector(_weighted_average([item.vector for item in candidates], [item.priority + 0.05 for item in candidates]))
        transformed = _normalize_vector([
            (1.0 - self.delta_scale) * a + self.delta_scale * b
            for a, b in zip(anchor_vector, assoc_vector)
        ])
        restored = _normalize_vector([
            (1.0 - self.anchor_pullback) * t + self.anchor_pullback * a
            for t, a in zip(transformed, anchor_vector)
        ])

        anchor_similarity = _cosine(anchor_vector, restored)
        drift = max(0.0, 1.0 - anchor_similarity)
        associated_texts = [_compress_text(item.text) for item in candidates[: self.creative_depth + 1]]
        lexical_novelty = 0.0
        if associated_texts:
            lexical_novelty = sum(_lexical_distance(anchor_text, text) for text in associated_texts) / len(associated_texts)
        novelty = max(drift, lexical_novelty * max(0.5, self.delta_scale))
        hint = self._render_hint(anchor_text=anchor_text, associated_texts=associated_texts, spontaneous=spontaneous)
        repetition = _repetition_fraction(hint)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        accepted = (
            bool(hint)
            and novelty >= self.novelty_threshold
            and drift <= self.drift_threshold
            and repetition <= self.repetition_threshold
            and elapsed_ms <= self.latency_budget_ms
        )
        return ImaginationProposal(
            accepted=accepted,
            anchor_text=anchor_text,
            drift=round(drift, 6),
            novelty=round(novelty, 6),
            repetition=round(repetition, 6),
            hint=hint,
            associated_texts=associated_texts,
        )

    def _select_associative_memories(self, anchor_vector: List[float], memories: List[MemoryItem]) -> List[MemoryItem]:
        scored: List[tuple[float, MemoryItem]] = []
        for item in memories[:12]:
            similarity = _cosine(anchor_vector, item.vector)
            if similarity >= 0.94:
                continue
            # Aim for moderate adjacency: not identical, not unrelated.
            target_similarity = 0.58
            novelty_term = abs(similarity - target_similarity)
            assistant_penalty = 0.18 if item.source == "assistant" else 0.0
            score = (1.0 - novelty_term) + 0.18 * item.priority + 0.08 * item.novelty - assistant_penalty
            scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[: max(1, self.creative_depth + 1)]]

    def _render_hint(self, anchor_text: str, associated_texts: List[str], spontaneous: bool) -> str:
        if not associated_texts:
            return ""
        lead = associated_texts[0]
        tail = associated_texts[1] if len(associated_texts) > 1 else ""
        if spontaneous:
            if tail:
                return f"It keeps linking {lead} with {tail}, so I am holding those two strands nearby without forcing a reply."
            return f"It keeps circling back to {lead}, so I am keeping that strand available without forcing it into speech."
        if tail:
            return f"One useful leap is to connect {lead} with {tail} without letting go of the current thread."
        return f"One useful leap is to treat {lead} as the adjacent thread that can extend the current thought without derailing it."


class LVTCBackend:
    name = "lvtc-local"

    def __init__(self, codec: EventCodec, config: BackendConfig) -> None:
        self.codec = codec
        self.config = config
        self.planner = LVTCPlanner(
            codec=codec,
            delta_scale=config.delta_scale,
            creative_depth=config.creative_depth,
            anchor_pullback=config.anchor_pullback,
            novelty_threshold=config.novelty_threshold,
            drift_threshold=config.drift_threshold,
            repetition_threshold=config.repetition_threshold,
            latency_budget_ms=config.latency_budget_ms,
        )

    def generate(self, stimulus: str, memories: List[MemoryItem], style: str, spontaneous: bool = False) -> BackendResult:
        plan = self.planner.propose(stimulus=stimulus, memories=memories, spontaneous=spontaneous)
        if spontaneous:
            text = self._render_spontaneous(memories=memories, plan=plan, style=style)
            return BackendResult(text=text, backend_name=self.name, metadata={"imagination": plan.accepted})

        intent = _detect_intent(stimulus)
        text = self._render_grounded(stimulus=stimulus, memories=memories, style=style, intent=intent, plan=plan)
        return BackendResult(text=text, backend_name=self.name, metadata={"imagination": plan.accepted, "intent": intent})

    def _render_spontaneous(self, memories: List[MemoryItem], plan: ImaginationProposal, style: str) -> str:
        preferred = _preferred_memories(memories)
        strongest = _compress_text(preferred[0].text) if preferred else "the last strong impression"
        parts = [
            f"I'm still carrying {strongest}.",
            "The internal stance is steady rather than urgent.",
        ]
        if plan.accepted:
            parts.append(plan.hint)
        else:
            parts.append("Nothing has justified a wider associative leap yet, so I am keeping the thread anchored.")
        return " ".join(parts)

    def _render_grounded(
        self,
        stimulus: str,
        memories: List[MemoryItem],
        style: str,
        intent: str,
        plan: ImaginationProposal,
    ) -> str:
        cleaned = stimulus.strip()
        preferred = _preferred_memories(memories)
        strongest = _compress_text(preferred[0].text) if preferred else "nothing anchored strongly enough yet"
        focus = _extract_focus(cleaned) or _extract_focus(strongest) or "the current thread"
        second = _compress_text(preferred[1].text) if len(preferred) > 1 else ""

        if intent == "greeting":
            parts = [
                "Henlo. I'm online and tracking the thread with you.",
                f"Right now the strongest live strand is {strongest}.",
            ]
        elif intent == "status":
            parts = [
                "I'm steady.",
                "I keep the running thread, rank what feels salient, and answer from that rather than from blank politeness.",
            ]
        elif intent == "capability":
            parts = [
                "Yes — I do more than just notice things now.",
                "I keep event-coded memory, pull nearby context, and take at most one guarded imaginative sidestep when it looks useful.",
            ]
        elif intent == "identity":
            parts = [
                "I'm WonderBot: a continuously ticking agent shell with event-coded memory, resonance gating, and a controlled imagination branch.",
                f"At the moment I'm centered on {focus}.",
            ]
        elif intent == "design":
            parts = [
                f"My grounded read is that the main design pressure is {focus}.",
                f"The closest anchored precedent I can see is {strongest}.",
            ]
        elif intent == "question":
            parts = []
            if cleaned:
                parts.append(f"I heard: {cleaned}.")
            parts.extend([
                f"My grounded take is to stay close to {focus}.",
                f"The nearest anchored thread I have is {strongest}.",
            ])
        elif intent == "sensor":
            parts = [
                f"I picked up a salient live change: {strongest}.",
                "I'm storing it as a live thread and staying anchored unless it develops further.",
            ]
        else:
            parts = []
            if cleaned:
                parts.append(f"I heard: {cleaned}.")
            parts.extend([
                f"The salient thread looks like {focus}.",
                f"The nearest anchored memory is {strongest}.",
            ])

        if second and intent in {"design", "question", "statement"}:
            parts.append(f"A nearby supporting strand is {second}.")

        if plan.accepted and _should_show_imagination(intent=intent, stimulus=cleaned):
            parts.append(plan.hint)
        elif intent in {"design", "capability"}:
            parts.append("I can stay tightly grounded or push one step wider from here, but I will not drift off into permanent dream-state.")

        return " ".join(parts)


class HFBackend:
    name = "hf"

    def __init__(self, codec: EventCodec, config: BackendConfig) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "HuggingFace backend requested, but transformers/torch are not installed. "
                "Install with: pip install -e .[hf]"
            ) from exc

        self._torch = torch
        self.model_name = config.hf_model
        self.max_new_tokens = config.max_new_tokens
        self.temperature = config.temperature
        self.use_chat_template = bool(config.hf_use_chat_template)
        self.trust_remote_code = bool(config.hf_trust_remote_code)
        self.quantized_4bit = bool(config.hf_load_in_4bit)
        self.planner = LVTCPlanner(
            codec=codec,
            delta_scale=config.delta_scale,
            creative_depth=config.creative_depth,
            anchor_pullback=config.anchor_pullback,
            novelty_threshold=config.novelty_threshold,
            drift_threshold=config.drift_threshold,
            repetition_threshold=config.repetition_threshold,
            latency_budget_ms=config.latency_budget_ms,
        )
        self._device = choose_component_device(config.hf_device or 'auto', 'auto')
        self.resolved_device = self._device.resolved
        self.offload_dir = ensure_offload_dir(config.hf_offload_dir or 'state/offload')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = parse_torch_dtype(config.hf_torch_dtype)
        compute_dtype = parse_torch_dtype(config.hf_compute_dtype)
        model_kwargs = {'trust_remote_code': self.trust_remote_code}
        if dtype not in {None, 'auto'}:
            model_kwargs['torch_dtype'] = dtype
        if self.quantized_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError(
                    '4-bit HF backend requested, but bitsandbytes support is unavailable. '
                    'Install with: pip install -e .[server-hf]'
                ) from exc
            if compute_dtype in {None, 'auto'}:
                compute_dtype = torch.float16
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=config.hf_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=bool(config.hf_double_quant),
            )
        if config.hf_device_map:
            model_kwargs['device_map'] = config.hf_device_map
            model_kwargs['offload_folder'] = self.offload_dir
            model_kwargs['low_cpu_mem_usage'] = True
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            if self.resolved_device != 'cpu' and not self.quantized_4bit:
                self.model = self.model.to(self.resolved_device)
        self.model.eval()

    def generate(self, stimulus: str, memories: List[MemoryItem], style: str, spontaneous: bool = False) -> BackendResult:
        plan = self.planner.propose(stimulus=stimulus, memories=memories, spontaneous=spontaneous)
        prompt = _build_prompt(stimulus=stimulus, memories=memories, style=style, spontaneous=spontaneous, plan=plan)
        encoded = self._encode_prompt(prompt)
        if hasattr(self.model, 'device'):
            encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        with self._torch.inference_mode():
            output = self.model.generate(
                **encoded,
                do_sample=True,
                temperature=max(0.01, self.temperature),
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = output[0][encoded["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if not text:
            text = "I had the shape of a response, but the backend returned silence."
        return BackendResult(text=text, backend_name=self.name, metadata={"imagination": plan.accepted, 'model': self.model_name, 'quantized_4bit': self.quantized_4bit})

    def _encode_prompt(self, prompt: str):
        if self.use_chat_template and getattr(self.tokenizer, 'chat_template', None):
            messages = [
                {
                    'role': 'system',
                    'content': 'You are WonderBot: grounded, concise, warm, and helpful. Answer from the current thread and use at most one controlled imaginative step if it helps.',
                },
                {'role': 'user', 'content': prompt},
            ]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors='pt',
                    return_dict=True,
                )
            except TypeError:
                pass
        return self.tokenizer(prompt, return_tensors='pt')


def create_backend(config: BackendConfig, codec: EventCodec) -> TextBackend:
    normalized = config.kind.strip().lower()
    if normalized in {"lvtc", "local", "lvtc-local"}:
        return LVTCBackend(codec=codec, config=config)
    if normalized == "hf":
        return HFBackend(codec=codec, config=config)
    raise ValueError(f"Unsupported backend kind: {config.kind}")


def _build_prompt(
    stimulus: str,
    memories: List[MemoryItem],
    style: str,
    spontaneous: bool,
    plan: ImaginationProposal,
) -> str:
    memory_block = "\n".join(f"- {memory.text}" for memory in memories[:6]) or "- none"
    mode = "spontaneous reflection" if spontaneous else "response"
    imagination_block = "none"
    if plan.accepted:
        imagination_block = (
            f"Use at most one guarded imaginative step. Anchor text: {plan.anchor_text}. "
            f"Hint: {plan.hint}"
        )
    return (
        f"You are WonderBot. Style: {style}. Mode: {mode}.\n"
        f"Relevant memories:\n{memory_block}\n\n"
        f"Current stimulus: {stimulus or '[idle]'}\n"
        f"Imagination rule: {imagination_block}\n"
        "First answer groundedly. If you use the imaginative hint, keep it to one concise sentence and stay close to the anchor. "
        "Avoid repetition and avoid drifting away from the current thread.\nAssistant:"
    )


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could", "do", "for", "from", "get", "give",
    "go", "how", "i", "if", "in", "into", "is", "it", "just", "like", "me", "more", "my", "not", "now", "of", "on",
    "or", "our", "please", "so", "stuff", "than", "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "to", "too", "us", "was", "we", "what", "when", "where", "which", "who", "why", "with", "would", "you", "your",
    "fren",
}


def _detect_intent(text: str) -> str:
    lowered = text.lower().strip()
    if not lowered:
        return "statement"
    if re.search(r"\bhow are you\b|\bhow's it going\b|\bare you okay\b|\byou okay\b", lowered):
        return "status"
    if re.search(r"(?:^|\s)(hello|henlo|hi|hey|yo)(?:$|[!,.?\s])", lowered):
        return "greeting"
    if lowered.startswith(("camera sees", "microphone hears")):
        return "sensor"
    if re.search(r"\bwhat are you\b|\bwho are you\b", lowered):
        return "identity"
    if re.search(r"\b(can you hear me|did you hear|can you understand|can you follow)\b", lowered):
        return "question"
    if re.search(r"\bwhat can you\b|\bdo you do more than\b|\bcan you (?:help|build|make|explain|tell|show|remember|plan|dream)\b", lowered):
        return "capability"
    if any(word in lowered for word in ("design", "build", "architecture", "implement", "wire", "integrate", "engine", "imagination")):
        return "design"
    if lowered.endswith("?") or lowered.startswith(("why ", "how ", "what ", "which ", "when ")):
        return "question"
    return "statement"


def _should_show_imagination(intent: str, stimulus: str) -> bool:
    lowered = stimulus.lower()
    if intent in {"design", "question"}:
        return True
    triggers = ("imagine", "what if", "could", "might", "idea", "design", "build", "invent", "creative")
    return any(trigger in lowered for trigger in triggers)


def _extract_focus(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    content = [word for word in words if word not in _STOPWORDS and len(word) > 2]
    if not content:
        return ""
    if len(content) == 1:
        return content[0]
    return " ".join(content[:3])


def _lexical_distance(a: str, b: str) -> float:
    a_words = {word for word in re.findall(r"[A-Za-z0-9']+", a.lower()) if word not in _STOPWORDS and len(word) > 2}
    b_words = {word for word in re.findall(r"[A-Za-z0-9']+", b.lower()) if word not in _STOPWORDS and len(word) > 2}
    if not a_words or not b_words:
        return 0.0
    overlap = len(a_words & b_words)
    union = len(a_words | b_words)
    return max(0.0, 1.0 - (overlap / max(1, union)))


def _compress_text(text: str, max_words: int = 12) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    cleaned = cleaned.strip(" .,;:-")
    if not cleaned:
        return "something faint but relevant"
    words = cleaned.split()
    if len(words) <= max_words:
        return cleaned
    return " ".join(words[:max_words]) + " …"


def _preferred_memories(memories: Sequence[MemoryItem]) -> List[MemoryItem]:
    filtered = [
        item for item in memories
        if item.source != "assistant"
        and not item.source.startswith("self/")
        and not (item.text or "").lower().startswith("self/identity:")
        and not (item.text or "").lower().startswith("self/style:")
    ]
    non_assistant = [item for item in memories if item.source != "assistant"]
    return filtered or non_assistant or list(memories)


def _weighted_average(vectors: Sequence[Sequence[float]], weights: Sequence[float]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    total_weight = max(1e-9, sum(max(0.0, w) for w in weights))
    out = [0.0] * dim
    for vector, weight in zip(vectors, weights):
        clamped = max(0.0, weight)
        for i, value in enumerate(vector):
            out[i] += value * clamped
    return [value / total_weight for value in out]


def _normalize_vector(vector: Sequence[float]) -> List[float]:
    if not vector:
        return []
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return list(vector)
    return [value / norm for value in vector]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _repetition_fraction(text: str, n: int = 3) -> float:
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    total = len(grams)
    unique = len(set(grams))
    return max(0.0, 1.0 - (unique / total))
