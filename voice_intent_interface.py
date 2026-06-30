# voice_intent_interface.py
from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).with_name("prompts") / "worker_intent_system.md"
TASK_COMPLETION_KEYWORDS = ("끝", "완료", "다했", "다 했", "체결", "조립", "마무리", "오케이")
APPROVE_KEYWORDS = ("응", "어", "네", "예", "그래", "좋아", "오케이", "ok", "맞아", "해줘", "조정")
REJECT_KEYWORDS = ("아니", "아니요", "괜찮", "그대로", "하지마", "필요없", "됐어", "no", "노")


@dataclass
class LlmAdjustmentDecision:
    # LLM이 사용자 의도와 상태를 보고 조정 여부와 목표 어깨각을 판단한 결과를 담는다.
    action: str = "unknown"
    target_shoulder_angle_deg: float | None = None
    confidence: float = 0.0
    reason: str = ""
    raw_text: str = ""
    source: str = "llm"
    is_invalid: bool = False
    clarification_question: str = ""

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


def is_task_completion_input(key: int, voice_text: str | None) -> bool:
    # SPACE 키나 단순 완료 키워드가 들어왔는지 바로 판별한다.
    if key == ord(" "):
        return True
    return _has_any(_normalize(voice_text or ""), TASK_COMPLETION_KEYWORDS)


def parse_worker_adjustment_input(
    wait_start_time: float,
    key: int,
    voice_text: str | None,
    control_type: str,
    rule_parser: "RuleIntentParser",
    llm_parser: "LlmIntentParser | None" = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Worker 주도 조건에서 Y/N 키, rule 응답, LLM 응답을 한 곳에서 해석한다.
    elapsed_wait = time.time() - wait_start_time
    response = {
        "answered": False,
        "text": "",
        "action": "unknown",
        "elapsed_wait": elapsed_wait,
        "target_shoulder_angle_deg": None,
        "latency": 0.0,
        "is_invalid": False,
        "reason": "",
        "source": "none",
        "clarification_question": "",
    }

    manual_action = _manual_adjustment_action(key)
    if manual_action:
        response.update(
            {
                "answered": True,
                "text": f"[Manual] {manual_action} adjustment",
                "action": manual_action,
                "source": "manual",
                "reason": "matched manual key",
            }
        )
        return response

    if not voice_text:
        return response

    response["text"] = voice_text
    if control_type == "LLM" and llm_parser is not None:
        llm_start_time = time.time()
        llm_decision = llm_parser.parse(
            voice_text,
            context="adjustment_response",
            metadata=metadata,
        )
        response.update(
            {
                "action": llm_decision.action,
                "target_shoulder_angle_deg": llm_decision.target_shoulder_angle_deg,
                "latency": time.time() - llm_start_time,
                "is_invalid": llm_decision.is_invalid,
                "reason": llm_decision.reason,
                "source": llm_decision.source,
                "clarification_question": llm_decision.clarification_question,
            }
        )
        response["answered"] = llm_decision.action in ("approve", "reject", "adjust")
        return response

    action = rule_parser.parse(voice_text, context="adjustment_response")
    response.update(
        {
            "answered": action in ("approve", "reject"),
            "action": action,
            "source": "rule",
            "reason": "matched rule keyword" if action != "unknown" else "no rule matched",
            "is_invalid": action == "unknown",
        }
    )
    return response


def _manual_adjustment_action(key: int) -> str | None:
    # Worker 응답 대기 중 Y/N 키를 approve/reject로 바꾼다.
    if key in (ord("y"), ord("Y")):
        return "approve"
    if key in (ord("n"), ord("N")):
        return "reject"
    return None


def _normalize(text: str) -> str:
    # 키워드 매칭을 위해 소문자화하고 공백을 제거한다.
    return text.lower().replace(" ", "")


def _has_any(normalized: str, keywords: tuple[str, ...]) -> bool:
    # 정규화된 문장에 키워드 중 하나라도 포함되는지 확인한다.
    return any(keyword.lower().replace(" ", "") in normalized for keyword in keywords)


class RuleIntentParser:
    # 명확한 키워드만 LLM 없이 빠르게 완료/승인/거절로 분류한다.
    def parse(self, text: str | None, context: str = "any") -> str:
        # 한 문장을 rule 기반으로 complete/approve/reject/unknown 중 하나로 해석한다.
        normalized = _normalize(text or "")
        if not normalized:
            return "unknown"

        if context == "task_completion" and _has_any(normalized, TASK_COMPLETION_KEYWORDS):
            return "complete"

        if context == "adjustment_response" and _has_any(normalized, REJECT_KEYWORDS):
            return "reject"

        if context == "adjustment_response" and _has_any(normalized, APPROVE_KEYWORDS):
            return "approve"

        if context == "any" and _has_any(normalized, TASK_COMPLETION_KEYWORDS):
            return "complete"

        if context == "any" and _has_any(normalized, REJECT_KEYWORDS):
            return "reject"

        if context == "any" and _has_any(normalized, APPROVE_KEYWORDS):
            return "approve"

        return "unknown"


class LlmIntentParser:
    # LLM으로 작업자 의도와 상태를 해석해 조정 여부와 목표 어깨각을 구조화해서 받는다.
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
    ) -> LlmAdjustmentDecision:
        # 발화와 context를 LLM에 보내고 LlmAdjustmentDecision으로 변환한다.
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
            return LlmAdjustmentDecision(
                raw_text=raw_text,
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
    def _result_from_json(parsed: dict[str, Any], raw_text: str) -> LlmAdjustmentDecision:
        # LLM이 반환한 JSON dict에서 action과 목표 어깨각만 main에서 쓰기 쉽게 꺼낸다.
        return LlmAdjustmentDecision(
            action=str(parsed.get("action", "unknown")),
            target_shoulder_angle_deg=LlmIntentParser._optional_float(
                parsed.get("target_shoulder_angle_deg")
                or parsed.get("target_angle_deg")
                or parsed.get("shoulder_angle_deg")
            ),
            confidence=float(parsed.get("confidence", 0.0) or 0.0),
            is_invalid=bool(parsed.get("is_invalid", False)),
            clarification_question=str(parsed.get("clarification_question", "")),
            reason=str(parsed.get("reason", "")),
            raw_text=raw_text,
        )

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        # LLM이 숫자를 문자열로 줘도 목표 어깨각으로 쓸 수 있게 float로 변환한다.
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None