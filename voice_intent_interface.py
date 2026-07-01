# voice_intent_interface.py
from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

# 프롬프트 파일 경로 (메모로 주신 프롬프트를 이 파일 안에 저장해두시면 됩니다)
DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).with_name("prompts") / "worker_intent_system.md"

# Rule 기반 처리를 위한 아주 심플한 키워드 (조금/많이 등은 LLM이 처리하므로 Rule은 Y/N만)
TASK_COMPLETION_KEYWORDS = ("끝", "완료", "다했", "다 했", "체결", "조립", "마무리", "오케이")
APPROVE_KEYWORDS = ("응", "어", "네", "예", "그래", "좋아", "오케이", "ok", "맞아", "해줘", "조정")
REJECT_KEYWORDS = ("아니", "아니요", "괜찮", "그대로", "하지마", "필요없", "됐어", "no", "노", "멈춰", "아파", "위험")


@dataclass
class LlmAdjustmentDecision:
    # 💡 프롬프트의 JSON 출력 형식과 100% 일치하도록 구조화
    action: str = "unknown"
    target_shoulder_angle_deg: float | None = None
    confidence: float = 0.0
    is_invalid: bool = False
    clarification_question: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ==============================================================================
# TTS / STT Hardware Classes (메인 코드에서 사용 중이므로 원본 유지)
# ==============================================================================
class QueuedTtsSpeaker:
    def __init__(self) -> None:
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive(): return
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def speak(self, text: str) -> None:
        if not text: return
        self.start()
        self._queue.put(text)

    def stop(self) -> None:
        self._queue.put(None)

    def _worker(self) -> None:
        engine = None
        while True:
            text = self._queue.get()
            try:
                if text is None: return
                if engine is None:
                    import pyttsx3
                    engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except Exception:
                engine = None
            finally:
                self._queue.task_done()


class ContinuousSpeechRecognizer:
    def __init__(self, language="ko-KR", energy_threshold=300, dynamic_energy_threshold=False, pause_threshold=0.5, listen_timeout_sec=1.0, phrase_time_limit_sec=3.0, microphone_index=None, on_text=None) -> None:
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
        if self._thread and self._thread.is_alive(): return
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
        with self._lock: self._latest_text = text
        if self.on_text: self.on_text(text)

    def _worker(self) -> None:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        recognizer.energy_threshold, recognizer.dynamic_energy_threshold, recognizer.pause_threshold = self.energy_threshold, self.dynamic_energy_threshold, self.pause_threshold
        microphone = sr.Microphone(device_index=self.microphone_index)
        with microphone as source:
            while not self._stop_event.is_set():
                try:
                    audio = recognizer.listen(source, timeout=self.listen_timeout_sec, phrase_time_limit=self.phrase_time_limit_sec)
                    text = recognizer.recognize_google(audio, language=self.language)
                    self._set_latest(text)
                except Exception:
                    continue


# ==============================================================================
# Intent Parsers (우리의 프롬프트와 목적에 맞게 완전히 경량화됨)
# ==============================================================================
def _normalize(text: str) -> str:
    return text.lower().replace(" ", "")

def _has_any(normalized: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword.lower().replace(" ", "") in normalized for keyword in keywords)

# 💡 메인 코드에서 "완료"라고 말했는지 확인할 때 쓰는 함수 (이 부분 추가!)
def is_task_completion_input(text: str | None) -> bool:
    normalized = _normalize(text or "")
    return _has_any(normalized, TASK_COMPLETION_KEYWORDS)

class RuleIntentParser:
    # Worker + Rule 조건용 심플 파서 (방향/크기 분석 없이 명확한 승인/거절만 판단)
    def parse(self, text: str | None, context: str = "any") -> str:
        normalized = _normalize(text or "")
        if not normalized: return "unknown"
        if _has_any(normalized, TASK_COMPLETION_KEYWORDS): return "complete"
        if _has_any(normalized, REJECT_KEYWORDS): return "reject"
        if _has_any(normalized, APPROVE_KEYWORDS): return "approve"
        return "unknown"


class LlmIntentParser:
    def __init__(self, api_key: str, base_url: str | None = None, model: str = "llama-3.1-8b-instant", system_prompt_path: str | Path = DEFAULT_SYSTEM_PROMPT_PATH, temperature: float = 0.0) -> None:
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.system_prompt_path = Path(system_prompt_path)

    def parse(self, text: str | None, context: str, metadata: dict[str, Any]) -> LlmAdjustmentDecision:
        raw_text = text or ""
        # 파일에서 시스템 프롬프트 읽어오기
        system_prompt = self.system_prompt_path.read_text(encoding="utf-8")
        
        # LLM에게 보낼 JSON Payload 조립
        payload = {
            "context": context,
            "utterance": raw_text,
            "metadata": metadata
        }
        user_content = json.dumps(payload, ensure_ascii=False, indent=2)

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
            
            # JSON 결과를 데이터클래스에 맵핑
            return LlmAdjustmentDecision(
                action=str(parsed.get("action", "unknown")),
                target_shoulder_angle_deg=self._optional_float(parsed.get("target_shoulder_angle_deg")),
                confidence=float(parsed.get("confidence", 0.0) or 0.0),
                is_invalid=bool(parsed.get("is_invalid", False)),
                clarification_question=str(parsed.get("clarification_question", "")),
                reason=str(parsed.get("reason", ""))
            )
        except Exception as exc:
            return LlmAdjustmentDecision(action="unknown", is_invalid=True, reason=f"LLM Error: {exc}")

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None or value == "": return None
        try: return float(value)
        except (TypeError, ValueError): return None