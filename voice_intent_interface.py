# voice_intent_interface.py
from __future__ import annotations

import json
import queue
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).with_name("prompts") / "worker_intent_system.md"


@dataclass
class IntentResult:
    # 음성/텍스트 발화 해석 결과를 main에서 쓰기 쉬운 공통 형태로 담는다.
    action: str = "unknown"
    direction: str = "none"
    amount: str = "none"
    confidence: float = 0.0
    normalized_intent: str = ""
    is_invalid: bool = False
    is_emergency_stop: bool = False
    clarification_question: str = ""
    reason: str = ""
    raw_text: str = ""
    source: str = "rule"

    def to_dict(self) -> dict[str, Any]:
        # CSV 기록이나 디버깅용으로 dataclass를 dict로 변환한다.
        return asdict(self)


class QueuedTtsSpeaker:
    # TTS 요청을 큐에 쌓아 main loop를 막지 않고 순차적으로 말하게 한다.
    def __init__(self) -> None:
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        # TTS worker thread가 없을 때만 새로 시작한다.
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def speak(self, text: str) -> None:
        # 말할 문장을 큐에 추가한다.
        if not text:
            return
        self.start()
        self._queue.put(text)

    def stop(self) -> None:
        # TTS worker thread에 종료 신호를 보낸다.
        self._queue.put(None)

    def _worker(self) -> None:
        # pyttsx3 엔진을 유지하면서 큐에 들어온 문장을 읽는다.
        engine = None
        while True:
            text = self._queue.get()
            try:
                if text is None:
                    return
                if engine is None:
                    import pyttsx3

                    engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except Exception:
                try:
                    if engine is not None:
                        engine.stop()
                except Exception:
                    pass
                engine = None
            finally:
                self._queue.task_done()


class ContinuousSpeechRecognizer:
    # 마이크를 백그라운드에서 계속 듣고, 가장 최근 인식 문장을 보관한다.
    def __init__(
        self,
        language: str = "ko-KR",
        energy_threshold: int = 300,
        dynamic_energy_threshold: bool = False,
        pause_threshold: float = 0.5,
        listen_timeout_sec: float = 1.0,
        phrase_time_limit_sec: float = 3.0,
        microphone_index: int | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> None:
        self.language = language
        self.energy_threshold = energy_threshold
        self.dynamic_energy_threshold = dynamic_energy_threshold
        self.pause_threshold = pause_threshold
        self.listen_timeout_sec = listen_timeout_sec
        self.phrase_time_limit_sec = phrase_time_limit_sec
        self.microphone_index = microphone_index
        self.on_text = on_text

        self._latest_text: str | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        # 음성 인식 thread를 시작한다.
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        # 음성 인식 thread 루프를 멈추도록 신호를 보낸다.
        self._stop_event.set()

    def get_and_clear(self) -> str | None:
        # 가장 최근 음성 인식 결과를 가져오고 내부 버퍼를 비운다.
        with self._lock:
            text = self._latest_text
            self._latest_text = None
        return text

    def _set_latest(self, text: str) -> None:
        # 인식된 문장을 저장하고 필요하면 callback에 넘긴다.
        with self._lock:
            self._latest_text = text
        if self.on_text:
            self.on_text(text)

    def _worker(self) -> None:
        # speech_recognition으로 짧은 발화를 반복해서 인식한다.
        import speech_recognition as sr

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = self.energy_threshold
        recognizer.dynamic_energy_threshold = self.dynamic_energy_threshold
        recognizer.pause_threshold = self.pause_threshold

        microphone = sr.Microphone(device_index=self.microphone_index)
        with microphone as source:
            while not self._stop_event.is_set():
                try:
                    audio = recognizer.listen(
                        source,
                        timeout=self.listen_timeout_sec,
                        phrase_time_limit=self.phrase_time_limit_sec,
                    )
                    text = recognizer.recognize_google(audio, language=self.language)
                    self._set_latest(text)
                except sr.WaitTimeoutError:
                    continue
                except Exception:
                    continue


class RuleIntentParser:
    # 명확한 키워드는 LLM 없이 빠르게 의도(action/direction/amount)로 분류한다.
    EARLY_STOP_KEYWORDS = (
        "그만",
        "중단",
        "포기",
        "아파",
        "위험",
        "멈춰",
        "스톱",
        "stop",
        "못하겠",
        "힘들",
        "실험종료",
    )
    COMPLETE_KEYWORDS = ("끝", "완료", "다했", "다 했", "체결", "조립", "마무리", "오케이")
    APPROVE_KEYWORDS = ("응", "어", "네", "예", "그래", "좋아", "오케이", "ok", "맞아", "해줘", "조정")
    REJECT_KEYWORDS = ("아니", "아니요", "괜찮", "그대로", "하지마", "필요없", "됐어", "no", "노")
    UP_KEYWORDS = ("올려", "높여", "위로", "상향")
    DOWN_KEYWORDS = ("내려", "낮춰", "아래", "하향")
    SMALL_KEYWORDS = ("조금", "살짝", "약간")
    LARGE_KEYWORDS = ("많이", "크게", "확")

    def parse(self, text: str | None, context: str = "any") -> IntentResult:
        # 한 문장을 rule 기반으로 early_stop/complete/approve/reject/adjust 중 하나로 해석한다.
        raw_text = text or ""
        normalized = self._normalize(raw_text)
        if not normalized:
            return IntentResult(raw_text=raw_text, is_invalid=True, reason="empty input")

        if self._has_any(normalized, self.EARLY_STOP_KEYWORDS):
            return IntentResult(
                action="early_stop",
                confidence=0.95,
                normalized_intent="작업 또는 실험 중단 요청",
                is_emergency_stop=True,
                raw_text=raw_text,
                reason="matched early stop keyword",
            )

        if context == "task_completion" and self._has_any(normalized, self.COMPLETE_KEYWORDS):
            return IntentResult(
                action="complete",
                confidence=0.85,
                normalized_intent="현재 작업 완료",
                raw_text=raw_text,
                reason="matched completion keyword",
            )

        direction = self._detect_direction(normalized)
        if direction != "none":
            return IntentResult(
                action="adjust",
                direction=direction,
                amount=self._detect_amount(normalized),
                confidence=0.85,
                normalized_intent="높이 조정 요청",
                raw_text=raw_text,
                reason="matched adjustment keyword",
            )

        if self._has_any(normalized, self.REJECT_KEYWORDS):
            return IntentResult(
                action="reject",
                direction="keep",
                confidence=0.85,
                normalized_intent="조정 거절 또는 현재 높이 유지",
                raw_text=raw_text,
                reason="matched rejection keyword",
            )

        if self._has_any(normalized, self.APPROVE_KEYWORDS):
            return IntentResult(
                action="approve",
                confidence=0.75,
                normalized_intent="조정 승인",
                raw_text=raw_text,
                reason="matched approval keyword",
            )

        if context == "any" and self._has_any(normalized, self.COMPLETE_KEYWORDS):
            return IntentResult(
                action="complete",
                confidence=0.85,
                normalized_intent="현재 작업 완료",
                raw_text=raw_text,
                reason="matched completion keyword",
            )

        return IntentResult(raw_text=raw_text, is_invalid=True, reason="no rule matched")

    def _detect_direction(self, normalized: str) -> str:
        # 높이 조정 방향을 up/down/none으로 감지한다.
        has_up = self._has_any(normalized, self.UP_KEYWORDS)
        has_down = self._has_any(normalized, self.DOWN_KEYWORDS)
        if has_up and not has_down:
            return "up"
        if has_down and not has_up:
            return "down"
        return "none"

    def _detect_amount(self, normalized: str) -> str:
        # 조정 크기를 small/medium/large로 감지한다.
        if self._has_any(normalized, self.LARGE_KEYWORDS):
            return "large"
        if self._has_any(normalized, self.SMALL_KEYWORDS):
            return "small"
        return "medium"

    @staticmethod
    def _normalize(text: str) -> str:
        # 키워드 매칭을 위해 소문자화하고 공백을 제거한다.
        return text.lower().replace(" ", "")

    @staticmethod
    def _has_any(normalized: str, keywords: tuple[str, ...]) -> bool:
        # 정규화된 문장에 키워드 중 하나라도 포함되는지 확인한다.
        return any(keyword.lower().replace(" ", "") in normalized for keyword in keywords)


class LlmIntentParser:
    # rule로 애매한 발화를 LLM에 보내 의도만 구조화해서 받는다.
    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "llama-3.1-8b-instant",
        system_prompt_path: str | Path = DEFAULT_SYSTEM_PROMPT_PATH,
        temperature: float = 0.0,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.system_prompt_path = Path(system_prompt_path)

    def parse(
        self,
        text: str | None,
        context: str = "any",
        metadata: dict[str, Any] | None = None,
    ) -> IntentResult:
        # 발화와 context를 LLM에 보내고 IntentResult로 변환한다.
        raw_text = text or ""
        system_prompt = self.system_prompt_path.read_text(encoding="utf-8")
        user_content = self._build_user_content(raw_text, context, metadata)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content.strip())
            return self._result_from_json(parsed, raw_text)
        except Exception as exc:
            return IntentResult(
                raw_text=raw_text,
                source="llm",
                is_invalid=True,
                reason=f"LLM intent parsing failed: {exc}",
            )

    def _build_user_content(
        self,
        text: str,
        context: str,
        metadata: dict[str, Any] | None,
    ) -> str:
        # LLM user message를 JSON 문자열로 구성한다.
        payload = {
            "context": context,
            "utterance": text,
            "metadata": metadata or {},
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    def _result_from_json(parsed: dict[str, Any], raw_text: str) -> IntentResult:
        # LLM이 반환한 JSON dict를 IntentResult로 변환한다.
        return IntentResult(
            action=str(parsed.get("action", "unknown")),
            direction=str(parsed.get("direction", "none")),
            amount=str(parsed.get("amount", "none")),
            confidence=float(parsed.get("confidence", 0.0) or 0.0),
            normalized_intent=str(parsed.get("normalized_intent", "")),
            is_invalid=bool(parsed.get("is_invalid", False)),
            is_emergency_stop=bool(parsed.get("is_emergency_stop", False)),
            clarification_question=str(parsed.get("clarification_question", "")),
            reason=str(parsed.get("reason", "")),
            raw_text=raw_text,
            source="llm",
        )
