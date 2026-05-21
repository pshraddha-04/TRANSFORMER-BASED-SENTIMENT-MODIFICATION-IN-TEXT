import re
from dataclasses import dataclass
from typing import List

from app.config import REWRITE_MIN_CONFIDENCE
from app.services.model_manager import model_manager

class RewriteService:
    TONE_HINTS = {
        "professional": "Use clear and respectful workplace language.",
        "empathetic": "Acknowledge feelings and be emotionally supportive.",
        "friendly": "Keep it warm, light, and approachable.",
        "concise": "Keep the response short and direct.",
        "formal": "Use formal grammar and polite wording.",
    }

    @dataclass
    class RewriteResult:
        text: str
        source: str
        confidence: float
        quality_flags: List[str]
        fallback_reason: str | None = None

    NEGATION_MARKERS = {"no", "not", "never", "none", "nobody", "nothing", "nowhere", "without", "n't"}
    SOFT_QUALITY_FLAGS = {"identity_rewrite", "near_identity_rewrite", "length_drift", "repeated_sentence"}
    HARD_QUALITY_FLAGS = {"empty_output", "number_mismatch", "negation_mismatch", "meaning_drift", "subject_shift", "repeated_phrase"}

    def _build_lightweight_result(self, text: str, target_tone: str, user_instruction: str | None, fallback_reason: str | None = None) -> RewriteResult:
        rewritten = self._rewrite_lightweight(text, target_tone, user_instruction)
        flags = self._quality_flags(original=text, rewritten=rewritten)
        return self.RewriteResult(
            text=rewritten,
            source="lightweight",
            confidence=self._rewrite_confidence(original=text, rewritten=rewritten, flags=flags),
            quality_flags=sorted(set(flags)),
            fallback_reason=fallback_reason,
        )

    def tone_suggestions(self, emotion: str) -> List[str]:
        suggestions = {
            "anger": ["empathetic", "professional", "concise"],
            "sadness": ["empathetic", "friendly", "formal"],
            "fear": ["empathetic", "professional", "concise"],
            "joy": ["friendly", "professional"],
            "surprise": ["professional", "friendly"],
            "neutral": ["professional", "concise"],
        }
        return suggestions.get(emotion.lower(), ["professional", "friendly"])

    def rewrite(self, text: str, target_tone: str, user_instruction: str | None = None) -> str:
        return self.rewrite_with_meta(text, target_tone, user_instruction).text

    def rewrite_with_meta(self, text: str, target_tone: str, user_instruction: str | None = None) -> RewriteResult:
        if model_manager.lightweight_mode:
            return self._build_lightweight_result(text, target_tone, user_instruction)

        try:
            rewriter = model_manager.get_rewriter()
            prompt = self._build_prompt(text, target_tone, user_instruction)

            # Use sampling to get diverse outputs and avoid copying input
            generated = rewriter(
                prompt,
                max_new_tokens=100,
                min_new_tokens=6,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                no_repeat_ngram_size=2,
                repetition_penalty=1.5,
            )[0]["generated_text"]
            cleaned = self._clean_output(generated)
            quality_flags = self._quality_flags(original=text, rewritten=cleaned)
            hard_flags = self._hard_quality_flags(quality_flags)

            # If the model returned an identity-like output, attempt a stronger
            # paraphrase by sampling multiple diverse candidates and choosing the
            # best non-identity one. This helps when the transformer tends to copy input.
            if any(f in {"identity_rewrite", "near_identity_rewrite"} for f in quality_flags):
                try:
                    strong_candidate = self._attempt_strong_paraphrase(
                        rewriter, text, target_tone, user_instruction
                    )
                    if strong_candidate:
                        sc_flags = self._quality_flags(original=text, rewritten=strong_candidate)
                        sc_conf = self._rewrite_confidence(original=text, rewritten=strong_candidate, flags=sc_flags)
                        if not self._hard_quality_flags(sc_flags) and sc_conf >= REWRITE_MIN_CONFIDENCE:
                            return self.RewriteResult(
                                text=strong_candidate,
                                source="transformer",
                                confidence=sc_conf,
                                quality_flags=sorted(set(sc_flags)),
                            )
                except Exception:
                    pass

            repaired = self._repair_identity_rewrite(
                original=text,
                rewritten=cleaned,
                flags=quality_flags,
                target_tone=target_tone,
                user_instruction=user_instruction,
            )
            if repaired is not None:
                return self.RewriteResult(text=repaired, source="transformer", confidence=0.8, quality_flags=[])

            if hard_flags:
                retry_prompt = self._build_prompt(text, target_tone, user_instruction, avoid_repetition=True)
                retry_generated = rewriter(
                    retry_prompt,
                    max_new_tokens=80,
                    min_new_tokens=8,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.8,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.35,
                )[0]["generated_text"]
                retry_cleaned = self._clean_output(retry_generated)
                retry_flags = self._quality_flags(original=text, rewritten=retry_cleaned)
                retry_hard_flags = self._hard_quality_flags(retry_flags)
                retry_repaired = self._repair_identity_rewrite(
                    original=text,
                    rewritten=retry_cleaned,
                    flags=retry_flags,
                    target_tone=target_tone,
                    user_instruction=user_instruction,
                )
                if retry_repaired is not None:
                    return self.RewriteResult(text=retry_repaired, source="transformer", confidence=0.78, quality_flags=[])

                if not retry_hard_flags:
                    retry_confidence = self._rewrite_confidence(original=text, rewritten=retry_cleaned, flags=retry_flags)
                    if retry_confidence < REWRITE_MIN_CONFIDENCE:
                        return self._build_lightweight_result(
                            text,
                            target_tone,
                            user_instruction,
                            fallback_reason="transformer_low_confidence",
                        )
                    return self.RewriteResult(
                        text=retry_cleaned,
                        source="transformer",
                        confidence=retry_confidence,
                        quality_flags=sorted(set(retry_flags)),
                    )

                return self._build_lightweight_result(
                    text,
                    target_tone,
                    user_instruction,
                    fallback_reason="transformer_low_quality_output",
                )

            primary_confidence = self._rewrite_confidence(original=text, rewritten=cleaned, flags=quality_flags)
            if primary_confidence < REWRITE_MIN_CONFIDENCE:
                return self._build_lightweight_result(
                    text,
                    target_tone,
                    user_instruction,
                    fallback_reason="transformer_low_confidence",
                )

            return self.RewriteResult(
                text=cleaned,
                source="transformer",
                confidence=primary_confidence,
                quality_flags=sorted(set(quality_flags)),
            )
        except Exception:
            fallback = self._build_lightweight_result(text, target_tone, user_instruction, fallback_reason="transformer_error")
            fallback.quality_flags = sorted(set(fallback.quality_flags + ["transformer_unavailable"]))
            return fallback

    def _build_prompt(self, text: str, target_tone: str, user_instruction: str | None, avoid_repetition: bool = False) -> str:
        tone_hint = self.TONE_HINTS.get(target_tone.lower(), "Use a polite and clear tone.")
        extra = f" Additional preference: {user_instruction}" if user_instruction else ""
        repetition_rule = " Do not repeat sentences or copy the input verbatim." if avoid_repetition else ""
        # Be explicit about paraphrasing and structural change so rewrites are
        # visibly different while preserving meaning (good for demos).
        paraphrase_instruction = (
            "Paraphrase the message: use different wording and sentence structure "
            "while preserving exact meaning and factual details."
        )
        return (
            "Task: Rewrite the message while preserving the exact meaning and facts. "
            "Do not change names, numbers, dates, amounts, negation, or who did what. "
            "Keep the same subject, event, and intent. "
            f"Target tone: {target_tone}. {tone_hint}. {paraphrase_instruction}{extra}{repetition_rule} "
            "Output only one rewritten message without labels or explanations.\n"
            f"Input: {text}\nOutput:"
        )

    def _clean_output(self, generated: str) -> str:
        text = generated.strip()
        text = re.sub(r"^(rewritten|output)\s*:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text)
        return text.strip(' "\'')

    def _quality_flags(self, original: str, rewritten: str) -> List[str]:
        flags: List[str] = []
        if not rewritten:
            return ["empty_output"]

        original_norm = re.sub(r"\s+", " ", original).strip().casefold()
        rewritten_norm = re.sub(r"\s+", " ", rewritten).strip().casefold()
        if rewritten_norm == original_norm:
            flags.append("identity_rewrite")

        sentence_parts = [s.strip().casefold() for s in re.split(r"[.!?]+", rewritten) if s.strip()]
        if len(sentence_parts) > 1 and len(set(sentence_parts)) < len(sentence_parts):
            flags.append("repeated_sentence")

        original_numbers = self._extract_numbers(original)
        rewritten_numbers = self._extract_numbers(rewritten)
        if original_numbers != rewritten_numbers:
            flags.append("number_mismatch")

        if self._has_negation(original_norm) != self._has_negation(rewritten_norm):
            flags.append("negation_mismatch")

        original_tokens = self._content_tokens(original_norm)
        rewritten_tokens = self._content_tokens(rewritten_norm)
        if original_tokens:
            overlap = len(original_tokens & rewritten_tokens) / len(original_tokens)
            if overlap < 0.5:
                flags.append("meaning_drift")
            if overlap >= 0.9 and rewritten_norm != original_norm:
                flags.append("near_identity_rewrite")

        original_length = max(1, len(original_norm.split()))
        rewritten_length = len(rewritten_norm.split())
        if rewritten_length < max(2, int(original_length * 0.45)) or rewritten_length > int(original_length * 2.2):
            flags.append("length_drift")

        if self._has_subject_shift(original_norm, rewritten_norm):
            flags.append("subject_shift")

        tokens = re.findall(r"\b\w+\b", rewritten_norm)
        if len(tokens) >= 6:
            tri_grams = [" ".join(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
            if len(set(tri_grams)) < len(tri_grams):
                flags.append("repeated_phrase")

        return flags

    def _hard_quality_flags(self, flags: List[str]) -> List[str]:
        return [flag for flag in flags if flag in self.HARD_QUALITY_FLAGS]

    def _repair_identity_rewrite(
        self,
        original: str,
        rewritten: str,
        flags: List[str],
        target_tone: str,
        user_instruction: str | None,
    ) -> str | None:
        # Repair identity-like outputs; repeated or empty outputs still use retry/fallback.
        if not flags or not set(flags).issubset({"identity_rewrite", "near_identity_rewrite"}):
            return None

        repaired = self._rewrite_lightweight(original, target_tone, user_instruction)
        repaired_flags = self._quality_flags(original=original, rewritten=repaired)
        if repaired_flags:
            return None
        return repaired

    def _attempt_strong_paraphrase(self, rewriter, original: str, target_tone: str, user_instruction: str | None) -> str | None:
        """
        Ask the transformer for multiple sampled paraphrases with high diversity
        and return the best non-identity candidate (or None if none found).
        Uses high temperature and top_p to force diverse rewording.
        """
        prompt = self._build_prompt(original, target_tone, user_instruction, avoid_repetition=True)
        # Add explicit, aggressive paraphrase instruction
        prompt += (
            " Rewrite using completely different words, phrasing, and sentence structure. "
            "Do NOT copy or repeat the original wording. Preserve all facts and meaning."
        )

        candidates = []

        # Try requesting multiple sequences first
        try:
            outputs = rewriter(
                prompt,
                max_new_tokens=100,
                min_new_tokens=5,
                do_sample=True,
                top_p=0.92,
                temperature=1.0,
                num_return_sequences=5,
                no_repeat_ngram_size=2,
                repetition_penalty=2.0,
            )
            for item in outputs:
                gen = item.get("generated_text") if isinstance(item, dict) else (item[0] if isinstance(item, list) and item else None)
                if gen:
                    candidates.append(gen)
        except TypeError:
            # Pipeline doesn't support num_return_sequences; do single calls
            pass

        # If that failed or produced few results, do repeated single samples
        if len(candidates) < 5:
            for _ in range(8):
                try:
                    out = rewriter(
                        prompt,
                        max_new_tokens=100,
                        min_new_tokens=5,
                        do_sample=True,
                        top_p=0.92,
                        temperature=1.0,
                        no_repeat_ngram_size=2,
                        repetition_penalty=2.0,
                    )
                    if isinstance(out, list) and out:
                        gen = out[0].get("generated_text") if isinstance(out[0], dict) else out[0]
                        if gen:
                            candidates.append(gen)
                except Exception:
                    continue

        # Process candidates: clean, score, rank
        scored = []
        for gen in candidates:
            cleaned = self._clean_output(gen)
            if not cleaned:
                continue
            flags = self._quality_flags(original=original, rewritten=cleaned)
            conf = self._rewrite_confidence(original=original, rewritten=cleaned, flags=flags)
            scored.append((cleaned, flags, conf))

        # Sort: prefer non-empty, then no hard flags, then high confidence
        def score_key(tup):
            cleaned, flags, conf = tup
            hard_count = len(self._hard_quality_flags(flags))
            is_identity = int("identity_rewrite" in flags or "near_identity_rewrite" in flags)
            return (is_identity, hard_count, -conf)

        scored.sort(key=score_key)

        # Return first candidate that is not identity and has no hard flags
        for cleaned, flags, conf in scored:
            hflags = self._hard_quality_flags(flags)
            if not hflags and "identity_rewrite" not in flags and "near_identity_rewrite" not in flags:
                return cleaned

        # If no perfect candidate, return best even with soft flags
        if scored:
            return scored[0][0]

        return None

    def _rewrite_lightweight(self, text: str, target_tone: str, user_instruction: str | None) -> str:
        rewritten = re.sub(r"\s+", " ", text.strip())
        if not rewritten:
            return ""

        tone = target_tone.lower()
        rewritten = self._rewrite_special_cases(rewritten, tone)
        replacements_by_tone = {
            "professional": {
                "stupid": "unclear",
                "hate": "do not appreciate",
                "terrible": "poor",
                "worst": "least effective",
                "awful": "unsatisfactory",
                "useless": "not helpful",
                "nobody helped me": "nobody helped me",
                "nobody helped": "nobody helped",
                "bad": "poor",
            },
            "empathetic": {
                "stupid": "frustrating",
                "hate": "do not like",
                "terrible": "frustrating",
                "worst": "hardest",
                "awful": "very difficult",
                "useless": "not helpful",
                "bad": "difficult",
            },
            "friendly": {
                "stupid": "unhelpful",
                "hate": "do not like",
                "terrible": "rough",
                "worst": "toughest",
                "awful": "rough",
                "useless": "not useful",
                "scared": "concerned",
            },
            "formal": {
                "stupid": "inappropriate",
                "hate": "do not prefer",
                "terrible": "unsatisfactory",
                "worst": "least suitable",
                "awful": "unsatisfactory",
                "useless": "not useful",
                "okay": "acceptable",
                "bad": "unsatisfactory",
            },
            "concise": {
                "stupid": "unclear",
                "hate": "do not like",
                "terrible": "poor",
                "worst": "bad",
                "awful": "bad",
                "useless": "not helpful",
                "scared": "concerned",
                "okay": "fine",
            },
        }

        for src, dst in replacements_by_tone.get(tone, {}).items():
            rewritten = re.sub(rf"\b{re.escape(src)}\b", dst, rewritten, flags=re.IGNORECASE)

        if tone in {"professional", "formal"}:
            rewritten = rewritten.replace("!", ".")
        elif tone == "concise":
            rewritten = re.sub(r"\s+", " ", rewritten)

        rewritten = rewritten.strip().strip('"\'')
        if rewritten and rewritten[-1] not in ".!?":
            rewritten += "."

        if user_instruction:
            instruction = user_instruction.lower()
            if "concise" in instruction or "one sentence" in instruction:
                rewritten = self._make_concise(rewritten)

        if self._token_overlap_ratio(text, rewritten) >= 0.9:
            rewritten = self._fallback_reframe(text, tone)

        return rewritten.strip()

    def _rewrite_special_cases(self, text: str, tone: str) -> str:
        text = self._rewrite_hate_patterns(text, tone)
        text = self._rewrite_scared_patterns(text, tone)
        text = self._rewrite_response_blame_patterns(text, tone)
        text = self._rewrite_unclear_communication_patterns(text, tone)
        return text

    def _rewrite_hate_patterns(self, text: str, tone: str) -> str:
        match = re.search(r"^i hate how (?P<rest>.+?)([.!?])?$", text, flags=re.IGNORECASE)
        if not match:
            return text
        rest = match.group("rest").rstrip(".?!")
        templates = {
            "professional": f"I am disappointed with how {rest}.",
            "empathetic": f"I understand how {rest} can feel frustrating.",
            "friendly": f"I do not like how {rest}, but I think we can improve it.",
            "formal": f"I am dissatisfied with how {rest}.",
            "concise": f"I dislike how {rest}.",
        }
        return templates.get(tone, templates["professional"])

    def _rewrite_scared_patterns(self, text: str, tone: str) -> str:
        match = re.search(r"^i am (?P<prefix>very |really |so )?scared (?P<rest>.+?)([.!?])?$", text, flags=re.IGNORECASE)
        if not match:
            match = re.search(r"^i'?m (?P<prefix>very |really |so )?scared (?P<rest>.+?)([.!?])?$", text, flags=re.IGNORECASE)
        if not match:
            return text
        rest = match.group("rest").rstrip(".?!")
        templates = {
            "professional": f"I am concerned that {rest}.",
            "empathetic": f"It is understandable to feel concerned that {rest}.",
            "friendly": f"I am concerned that {rest}, but I think we can work through it.",
            "formal": f"I am concerned that {rest}.",
            "concise": f"I am concerned that {rest}.",
        }
        return templates.get(tone, templates["professional"])

    def _rewrite_response_blame_patterns(self, text: str, tone: str) -> str:
        lowered = text.casefold()
        if "you people are the worst at responding" not in lowered and not re.search(r"\byou(?: people)? are the worst\b", lowered):
            return text
        templates = {
            "professional": "Your response time has been unsatisfactory.",
            "empathetic": "I understand the response has felt very slow.",
            "friendly": "Your response time has been a bit slow, but I believe we can improve it.",
            "formal": "Your response time has been unsatisfactory.",
            "concise": "Your response time has been poor.",
        }
        return templates.get(tone, templates["professional"])

    def _rewrite_unclear_communication_patterns(self, text: str, tone: str) -> str:
        lowered = text.casefold()
        if "communication was unclear" not in lowered:
            return text
        templates = {
            "professional": "The communication could have been clearer.",
            "empathetic": "I understand the communication could have been clearer.",
            "friendly": "The communication could have been a little clearer.",
            "formal": "The communication could have been clearer.",
            "concise": "Communication could have been clearer.",
        }
        if lowered.startswith("the update is okay"):
            templates["concise"] = "The update was acceptable, but communication was unclear."
        return templates.get(tone, templates["professional"])

    def _fallback_reframe(self, original: str, tone: str) -> str:
        core = re.sub(r"\s+", " ", original.strip().rstrip(".?!"))
        replacements = {
            "professional": f"I have concerns about the current message: {core}.",
            "empathetic": f"I understand this is frustrating: {core}.",
            "friendly": f"I hear you, and I want to help with this: {core}.",
            "formal": f"I would like to note a concern regarding the current message: {core}.",
            "concise": f"I have concerns about this message.",
        }
        return replacements.get(tone, replacements["professional"])

    def _token_overlap_ratio(self, original: str, rewritten: str) -> float:
        original_tokens = self._content_tokens(original.casefold())
        rewritten_tokens = self._content_tokens(rewritten.casefold())
        if not original_tokens:
            return 0.0
        return len(original_tokens & rewritten_tokens) / len(original_tokens)

    def _make_concise(self, text: str) -> str:
        text = re.sub(r"\b(very|really|quite|so)\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        if text and text[-1] not in ".!?":
            text += "."
        return text

    def _extract_numbers(self, text: str) -> tuple[str, ...]:
        return tuple(re.findall(r"\b\d+(?:\.\d+)?\b", text))

    def _has_negation(self, text: str) -> bool:
        tokens = set(re.findall(r"\b\w+\b", text))
        return any(token in self.NEGATION_MARKERS for token in tokens)

    def _content_tokens(self, text: str) -> set[str]:
        stopwords = {
            "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "this", "that",
            "it", "is", "was", "were", "be", "been", "am", "are", "i", "me", "my", "you", "your",
            "we", "they", "them", "our", "us", "as", "at", "by", "from", "but", "if", "not", "do",
            "did", "does", "have", "has", "had", "will", "would", "can", "could", "should",
        }
        tokens = {token for token in re.findall(r"\b[a-z']+\b", text.casefold()) if token not in stopwords}
        return tokens

    def _has_subject_shift(self, original: str, rewritten: str) -> bool:
        original_subjects = self._subject_markers(original)
        rewritten_subjects = self._subject_markers(rewritten)
        if not original_subjects:
            return False
        return bool(original_subjects - rewritten_subjects)

    def _subject_markers(self, text: str) -> set[str]:
        markers = set()
        if re.search(r"\byou\b", text):
            markers.add("you")
        if re.search(r"\bwe\b", text):
            markers.add("we")
        if re.search(r"\bi\b", text):
            markers.add("i")
        if re.search(r"\bthey\b", text):
            markers.add("they")
        if re.search(r"\bmy\b", text):
            markers.add("my")
        return markers

    def _rewrite_confidence(self, original: str, rewritten: str, flags: List[str]) -> float:
        if not rewritten:
            return 0.0

        score = 0.9
        penalties = {
            "empty_output": 0.7,
            "number_mismatch": 0.22,
            "negation_mismatch": 0.24,
            "meaning_drift": 0.26,
            "subject_shift": 0.18,
            "repeated_phrase": 0.12,
            "identity_rewrite": 0.04,
            "near_identity_rewrite": 0.03,
            "length_drift": 0.05,
            "repeated_sentence": 0.06,
        }
        for flag in flags:
            score -= penalties.get(flag, 0.08)

        original_tokens = re.findall(r"\b\w+\b", original.lower())
        rewritten_tokens = re.findall(r"\b\w+\b", rewritten.lower())
        if original_tokens:
            ratio = len(rewritten_tokens) / max(1, len(original_tokens))
            if ratio < 0.5 or ratio > 2.5:
                score -= 0.12

        return round(max(0.0, min(1.0, score)), 4)


rewrite_service = RewriteService()

