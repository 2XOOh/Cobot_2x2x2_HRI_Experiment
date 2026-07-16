from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


DEFAULT_INTENT_PROMPT_PATH = Path(__file__).with_name("prompts") / "intent_interpreter_system.md"

ACTION_COMPLETE = "complete"
ACTION_ADJUST = "adjust"
ACTION_KEEP = "keep"
ACTION_CLARIFY = "clarify"
ACTION_NONE = "none"

VALID_ACTIONS = {
    ACTION_COMPLETE,
    ACTION_ADJUST,
    ACTION_KEEP,
    ACTION_CLARIFY,
    ACTION_NONE,
}

DIRECTION_UP = "up"
DIRECTION_DOWN = "down"
DIRECTION_NONE = "none"
VALID_DIRECTIONS = {DIRECTION_UP, DIRECTION_DOWN, DIRECTION_NONE}

AMOUNT_SMALL = 0.33
AMOUNT_NORMAL = 0.66
AMOUNT_LARGE = 1.0
VALID_AMOUNT_RATIOS = {AMOUNT_SMALL, AMOUNT_NORMAL, AMOUNT_LARGE}


@dataclass
class IntentResult:
    action: str = ACTION_NONE
    direction: str = DIRECTION_NONE
    amount_ratio: float | None = None
    target_shoulder_angle_deg: float | None = None
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "direction": self.direction,
            "amount_ratio": self.amount_ratio,
            "target_shoulder_angle_deg": self.target_shoulder_angle_deg,
            "confidence": self.confidence,
            "reason": self.reason,
        }


def manual_task_completion_from_key(key: int) -> IntentResult | None:
    if key == ord(" "):
        return IntentResult(
            action=ACTION_COMPLETE,
            confidence=1.0,
            reason="SPACE",
        )
    return None


def manual_worker_answer_from_key(key: int) -> IntentResult | None:
    if key in (ord("n"), ord("N")):
        return IntentResult(
            action=ACTION_KEEP,
            confidence=1.0,
            reason="N key",
        )
    if key in (ord("y"), ord("Y")):
        return IntentResult(
            action=ACTION_CLARIFY,
            confidence=1.0,
            reason="Y key",
        )
    return None


class QueuedTtsSpeaker:
    def __init__(self) -> None:
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def speak(self, text: str) -> None:
        if not text:
            return
        self.start()
        self._queue.put(text)

    def stop(self) -> None:
        self._queue.put(None)

    def wait_until_done(self, timeout_sec: float | None = None) -> None:
        deadline = None if timeout_sec is None else time.time() + timeout_sec
        while self._queue.unfinished_tasks:
            if deadline is not None and time.time() >= deadline:
                return
            time.sleep(0.05)

    def _worker(self) -> None:
        while True:
            text = self._queue.get()
            engine = None
            try:
                if text is None:
                    return

                import pyttsx3

                engine = pyttsx3.init()
                engine.setProperty("rate", 160)
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:
                print(f"[TTS ERROR] {exc}")
            finally:
                try:
                    if engine is not None:
                        engine.stop()
                except Exception:
                    pass
                self._queue.task_done()


class ContinuousSpeechRecognizer:
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
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def get_and_clear(self) -> str | None:
        with self._lock:
            text = self._latest_text
            self._latest_text = None
        return text

    def _set_latest(self, text: str) -> None:
        with self._lock:
            self._latest_text = text
        if self.on_text:
            self.on_text(text)

    def _worker(self) -> None:
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


class LlmIntentInterpreter:
    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "llama-3.1-8b-instant",
        system_prompt_path: str | Path = DEFAULT_INTENT_PROMPT_PATH,
        temperature: float = 0.0,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.system_prompt_path = Path(system_prompt_path)
        self.last_latency_s = 0.0
        self.last_request_payload: dict[str, Any] | None = None
        self.last_raw_response: str | None = None
        self.last_parsed_response: dict[str, Any] | None = None
        self.last_error: str | None = None

    # 작업 중 나온 음성 문장이 "작업 완료"인지 해석한다.
    def interpret_task_completion(self, text: str | None) -> IntentResult:
        utterance = (text or "").strip()
        if not utterance:
            return IntentResult(action=ACTION_NONE, reason="empty utterance")

        return self._request_intent(
            {
                "task": "task_completion",
                "utterance": utterance,
            }
        )

    # 리뷰 단계에서 시스템 판단 또는 작업자 답변을 조정 의도로 해석한다.
    def interpret_adjustment(
        self,
        text: str | None,
        cycle_result: Any,
        safe_min_shoulder_angle_deg: float,
        default_safe_shoulder_angle_deg: float,
        safe_max_shoulder_angle_deg: float,
        is_first_completed_cycle: bool = False,
        is_system_review: bool = False,
    ) -> IntentResult:
        utterance = (text or "").strip()
        if not utterance and not is_system_review:
            return IntentResult(action=ACTION_NONE, reason="empty utterance")

        return self._request_intent(
            {
                "task": "adjustment",
                "utterance": utterance,
                "is_first_completed_cycle": bool(is_first_completed_cycle),
                "cycle_result": {
                    "is_risky_cycle": bool(cycle_result.is_risky_cycle),
                    "representative_shoulder_angle_deg": float(cycle_result.representative_shoulder_angle_deg),
                },
                "safe_angle_range": {
                    "min": float(safe_min_shoulder_angle_deg),
                    "default": float(default_safe_shoulder_angle_deg),
                    "max": float(safe_max_shoulder_angle_deg),
                },
            }
        )

    # 공통 LLM 호출 함수: payload를 보내고 응답 JSON을 IntentResult로 바꾼다.
    def _request_intent(self, payload: dict[str, Any]) -> IntentResult:
        started_at = time.time()
        self.last_request_payload = payload
        self.last_raw_response = None
        self.last_parsed_response = None
        self.last_error = None
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt_path.read_text(encoding="utf-8")},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
                ],
                temperature=self.temperature,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            self.last_raw_response = response.choices[0].message.content or "{}"
            parsed = json.loads(self.last_raw_response.strip())
            self.last_parsed_response = parsed
            return intent_from_json(parsed)
        except Exception as exc:
            self.last_error = str(exc)
            return IntentResult(
                action=ACTION_CLARIFY,
                confidence=0.0,
                reason=f"LLM intent failed: {exc}",
            )
        finally:
            self.last_latency_s = time.time() - started_at


def intent_from_json(parsed: dict[str, Any]) -> IntentResult:
    return IntentResult(
        action=str(parsed.get("action", ACTION_CLARIFY)).strip().lower(),
        direction=str(parsed.get("direction", DIRECTION_NONE)).strip().lower(),
        amount_ratio=_optional_float(parsed.get("amount_ratio")),
        target_shoulder_angle_deg=_optional_float(parsed.get("target_shoulder_angle_deg")),
        confidence=_float(parsed.get("confidence"), 0.0),
        reason=str(parsed.get("reason", "")),
    )


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
