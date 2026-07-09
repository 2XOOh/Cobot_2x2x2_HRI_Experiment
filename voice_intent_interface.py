# voice_intent_interface.py
from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).with_name("prompts") / "worker_intent_runtime.md"
# The keyword lists below are used by RuleIntentParser and simple non-LLM checks.
# Worker + LLM conditions send the full utterance to the LLM prompt instead.
TASK_COMPLETION_KEYWORDS = (
    "끝",
    "종료",
    "완료",
    "다했",
    "다 했",
    "다됐",
    "다 됐",
    "끝냈",
    "끝났",
    "마쳤",
    "마무리",
    "다뺐",
    "다 뺐",
    "다풀었",
    "다 풀었",
    "finished",
    "done",
)
APPROVE_KEYWORDS = ("응", "어", "네", "예", "그래", "좋아", "오케이", "ok", "맞아", "해줘", "조정")
REJECT_KEYWORDS = ("아니", "아니요", "괜찮", "그대로", "하지마", "필요없", "됐어", "no", "노")
UPWARD_ADJUST_KEYWORDS = ("올려", "올리", "높여", "위로", "높게")
DOWNWARD_ADJUST_KEYWORDS = ("낮춰", "낮추", "내려", "내리", "아래로", "낮게")
LLM_ACTION_CONFIDENCE_THRESHOLD = 0.75
RULE_WORKER_UP_STEP_DEG = 10.0
RULE_WORKER_DOWN_STEP_DEG = 10.0
RULE_MIN_ANGLE_TOLERANCE_DEG = 1.0
AMBIGUOUS_ASR_PHRASES = (
    "알겠",
    "알았",
    "알아먹",
    "알아들",
    "다시해",
    "그렇게해",
    "그걸로해",
    "그냥해",
)
DIRECTIONLESS_ADJUSTMENT_HINTS = ("조정", "변경", "바꿔", "바꾸")
DIRECTION_OR_MAINTAIN_HINTS = (
    *UPWARD_ADJUST_KEYWORDS,
    *DOWNWARD_ADJUST_KEYWORDS,
    *REJECT_KEYWORDS,
    "유지",
    "이대로",
    "현재",
    "위쪽",
    "아래쪽",
)
IGNORED_TTS_ECHO_PHRASES = (
    "인식하지못",
    "다시말씀",
    "유지할지올릴지내릴지",
    "높이를유지할지",
    "불편자세를감지",
    "안전자세가감지",
    "작업높이를변경해드릴까요",
    "작업높이를변경할까요",
    "현재로봇이전달할수있는최저높이",
    "더낮춰서전달할수없습니다",
    "현재로봇이전달할수있는최고높이",
    "더높여서전달할수없습니다",
)
DIRECTIONLESS_CONFIRMATION_PHRASES = (
    "응",
    "어",
    "네",
    "예",
    "그래",
    "좋아",
    "오케이",
    "ok",
    "맞아",
    "해줘",
    "해주세요",
    "조정해줘",
    "조정해주세요",
    "바꿔줘",
    "바꿔주세요",
    "변경해줘",
    "변경해주세요",
    "알겠어",
    "알겠어요",
    "알겠습니다",
    "알았어",
    "알았어요",
    "알았습니다",
    "알아먹었어",
    "알아먹었어요",
    "알아들었어",
    "알아들었어요",
)
DIRECTIONLESS_MODIFIER_PHRASES = (
    "조금",
    "조금만",
    "조금만 더",
    "살짝",
    "약간",
    "많이",
    "확",
    "더",
    "엄청",
    "엄청 조금만",
    "조금만 해줘",
    "살짝 해줘",
    "약간 해줘",
)
HEIGHT_POSTURE_HINTS = (
    "높이",
    "자세",
    "어깨",
    "팔",
    "불편",
    "부담",
    "편하",
    "편해",
    "위쪽",
    "위로",
    "아래",
    "낮",
    "높",
    "내려",
    "내리",
    "올려",
    "올리",
    "유지",
    "그대로",
    "괜찮",
    "필요없",
    "됐어",
    "조정",
    "변경",
    "바꿔",
)


@dataclass
class LlmAdjustmentDecision:
    # LLM이 사용자 의도와 상태를 보고 조정 여부와 목표 어깨각을 판단한 결과를 담는다.
    action: str = "unknown"
    direction: str = "unclear"
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

    def wait_until_done(self, timeout_sec: float | None = None) -> None:
        deadline = None if timeout_sec is None else time.time() + timeout_sec
        while self._queue.unfinished_tasks:
            if deadline is not None and time.time() >= deadline:
                return
            time.sleep(0.05)

    def _worker(self) -> None:
        # pyttsx3 엔진을 유지하면서 큐에 들어온 문장을 읽는다.
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


def is_task_completion_input(
    key: int,
    voice_text: str | None,
    llm_parser: "LlmIntentParser | None" = None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    # 명확한 완료 표현은 즉시 처리하고, 나머지는 LLM으로 종료 의미를 해석한다.
    if key == ord(" "):
        return True

    normalized = _normalize(voice_text or "")
    if not normalized:
        return False
    if _has_any(normalized, TASK_COMPLETION_KEYWORDS):
        return True
    if llm_parser is None:
        return False

    decision = llm_parser.parse(
        voice_text,
        context="task_completion",
        metadata=metadata,
    )
    if decision.source == "llm_error":
        print(f"[작업 완료 LLM 오류] {decision.reason}")
        return False
    return (
        decision.action == "complete"
        and decision.confidence >= LLM_ACTION_CONFIDENCE_THRESHOLD
        and not decision.is_invalid
    )


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
        "direction": "unclear",
        "elapsed_wait": elapsed_wait,
        "target_shoulder_angle_deg": None,
        "latency": 0.0,
        "confidence": 0.0,
        "is_invalid": False,
        "reason": "",
        "source": "none",
        "clarification_question": "",
    }

    manual_action = _manual_adjustment_action(key)
    if manual_action:
        response.update(
            {
                "answered": manual_action in ("reject", "adjust"),
                "text": f"[Manual] {manual_action} adjustment",
                "action": manual_action,
                "source": "manual",
                "reason": "matched manual key",
                "confidence": 1.0,
                "clarification_question": (
                    "높이를 유지할지, 올릴지, 내릴지 말씀해 주세요."
                    if manual_action == "ask_clarification"
                    else ""
                ),
            }
        )
        return response

    if not voice_text:
        return response

    response["text"] = voice_text
    if _is_tts_echo(voice_text):
        response.update(
            {
                "action": "ignored",
                "source": "ignored",
                "reason": "ignored likely TTS echo",
            }
        )
        return response

    if _is_directionless_modifier(_normalize(voice_text)):
        response.update(
            {
                "action": "ask_clarification",
                "direction": "unclear",
                "source": "semantic_guard",
                "confidence": 1.0,
                "is_invalid": False,
                "reason": "Adjustment strength was stated without an upward or downward direction.",
                "clarification_question": "높이를 유지할지, 올릴지, 내릴지 말씀해 주세요.",
            }
        )
        return response

    if control_type in ("LLM", "Rule"):
        if llm_parser is None:
            response.update(
                {
                    "is_invalid": True,
                    "reason": f"{control_type} worker response requires an LLM parser.",
                }
            )
            return response

        llm_start_time = time.time()
        llm_decision = llm_parser.parse(
            voice_text,
            context="adjustment_response",
            metadata=metadata,
        )
        response.update(
            {
                "action": llm_decision.action,
                "direction": llm_decision.direction,
                "target_shoulder_angle_deg": llm_decision.target_shoulder_angle_deg,
                "latency": time.time() - llm_start_time,
                "confidence": llm_decision.confidence,
                "is_invalid": llm_decision.is_invalid,
                "reason": llm_decision.reason,
                "source": llm_decision.source,
                "clarification_question": llm_decision.clarification_question,
            }
        )
        if llm_decision.source == "llm_error":
            return response

        if response["action"] == "ask_clarification" and not response["clarification_question"]:
            response["clarification_question"] = "높이를 유지할지, 올릴지, 내릴지 말씀해 주세요."

        retry_reason = _llm_worker_retry_reason(
            voice_text,
            response,
            metadata,
            validate_target=control_type == "LLM",
        )
        if retry_reason:
            response.update(
                {
                    "answered": False,
                    "action": "unknown",
                    "target_shoulder_angle_deg": None,
                    "is_invalid": True,
                    "reason": retry_reason,
                    "clarification_question": "",
                }
            )
            return response

        if control_type == "Rule":
            return _apply_rule_worker_policy(response, metadata)

        response["answered"] = llm_decision.action in ("approve", "reject", "adjust")
        return response

    response.update(
        {
            "is_invalid": True,
            "reason": f"Unsupported worker control type: {control_type}",
        }
    )
    return response


def _manual_adjustment_action(key: int) -> str | None:
    # Worker 응답 대기 중 Y/N 키를 approve/reject로 바꾼다.
    if key in (ord("y"), ord("Y")):
        return "ask_clarification"
    if key in (ord("n"), ord("N")):
        return "reject"
    return None


def resolve_rule_worker_target_angle_by_policy(
    metadata: dict[str, Any] | None,
    direction: str,
) -> float | None:
    if not isinstance(metadata, dict):
        return None

    try:
        current_angle = float(metadata["cycle_representative_shoulder_angle_deg"])
        effective_min = float(metadata["effective_min_shoulder_deg"])
        safe_max = float(metadata.get("pilot_functional_max_shoulder_deg", 80.0))
        up_step_deg = float(metadata.get("rule_worker_up_step_deg", RULE_WORKER_UP_STEP_DEG))
        down_step_deg = float(
            metadata.get("rule_worker_down_step_deg", RULE_WORKER_DOWN_STEP_DEG)
        )
        cycle_is_risky = bool(metadata.get("cycle_is_risky", False))
    except (KeyError, TypeError, ValueError):
        return None

    if direction == "up":
        return max(0.0, min(180.0, current_angle + up_step_deg))
    if direction == "down":
        if cycle_is_risky:
            return max(effective_min, safe_max)
        return max(0.0, min(180.0, current_angle - down_step_deg))
    return None


def _apply_rule_worker_policy(
    response: dict[str, Any],
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    llm_action = str(response.get("action", "unknown"))
    response["source"] = "rule_llm"

    if llm_action == "reject":
        response.update(
            {
                "answered": True,
                "target_shoulder_angle_deg": None,
                "reason": f"{response.get('reason', '')} Rule policy kept the current height.".strip(),
            }
        )
        return response

    direction = str(response.get("direction", "")).lower()
    if direction not in ("up", "down"):
        direction = _llm_adjustment_direction(response, metadata)
    if llm_action not in ("approve", "adjust") or direction is None:
        response.update(
            {
                "answered": False,
                "action": "unknown",
                "target_shoulder_angle_deg": None,
                "is_invalid": True,
                "reason": (
                    f"{response.get('reason', '')} "
                    "Rule policy could not determine an upward or downward direction."
                ).strip(),
            }
        )
        return response

    target_angle = resolve_rule_worker_target_angle_by_policy(metadata, direction)
    if target_angle is None:
        response.update(
            {
                "answered": False,
                "action": "unknown",
                "target_shoulder_angle_deg": None,
                "is_invalid": True,
                "reason": "Rule policy could not calculate a target shoulder angle.",
            }
        )
        return response

    policy_reason = (
        "Rule policy applied current shoulder angle +10 degrees."
        if direction == "up"
        else (
            "Rule policy guided the risky posture to the safe-range upper bound."
            if bool((metadata or {}).get("cycle_is_risky", False))
            else "Rule policy applied current shoulder angle -10 degrees."
        )
    )
    response.update(
        {
            "answered": True,
            "action": "adjust",
            "target_shoulder_angle_deg": target_angle,
            "is_invalid": False,
            "reason": f"{response.get('reason', '')} {policy_reason}".strip(),
        }
    )
    return response


def _llm_adjustment_direction(
    response: dict[str, Any],
    metadata: dict[str, Any] | None,
) -> str | None:
    if not isinstance(metadata, dict):
        return None
    try:
        current_angle = float(metadata["cycle_representative_shoulder_angle_deg"])
        llm_target = float(response["target_shoulder_angle_deg"])
    except (KeyError, TypeError, ValueError):
        return None

    if llm_target > current_angle:
        return "up"
    if llm_target < current_angle:
        return "down"
    try:
        effective_min = float(metadata["effective_min_shoulder_deg"])
    except (KeyError, TypeError, ValueError):
        return None
    if current_angle <= effective_min + RULE_MIN_ANGLE_TOLERANCE_DEG:
        return "down"
    return None


def _llm_worker_retry_reason(
    voice_text: str,
    response: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    validate_target: bool = True,
) -> str:
    action = str(response.get("action", "unknown"))
    if action not in ("approve", "reject", "adjust"):
        return ""

    try:
        confidence = float(response.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    if confidence < LLM_ACTION_CONFIDENCE_THRESHOLD:
        return (
            f"LLM confidence {confidence:.2f} is below "
            f"{LLM_ACTION_CONFIDENCE_THRESHOLD:.2f}; asking worker to repeat."
        )

    if (
        validate_target
        and action in ("approve", "adjust")
        and response.get("target_shoulder_angle_deg") is None
    ):
        return "LLM selected height adjustment without a target angle; asking worker to repeat."

    if action in ("approve", "adjust"):
        direction = str(response.get("direction", "")).lower()
        if direction not in ("up", "down"):
            return "LLM selected height adjustment without a clear direction."

    if validate_target and action in ("approve", "adjust"):
        try:
            current_worker_angle = float((metadata or {})["cycle_representative_shoulder_angle_deg"])
            current_robot_angle = float((metadata or {})["current_robot_shoulder_angle_deg"])
            target_angle = float(response["target_shoulder_angle_deg"])
            effective_min = float((metadata or {})["effective_min_shoulder_deg"])
            robot_max = float((metadata or {})["robot_max_reachable_shoulder_deg"])
        except (KeyError, TypeError, ValueError):
            return "Robot-relative shoulder-angle metadata is missing or invalid."

        tolerance_deg = RULE_MIN_ANGLE_TOLERANCE_DEG
        at_lower_limit = current_robot_angle <= effective_min + tolerance_deg
        at_upper_limit = current_robot_angle >= robot_max - tolerance_deg
        if direction == "up" and target_angle <= current_worker_angle:
            if not (at_upper_limit and target_angle >= current_worker_angle - tolerance_deg):
                return "Upward intent produced a non-upward target angle."
        if direction == "down" and target_angle >= current_worker_angle:
            if not (at_lower_limit and target_angle <= current_worker_angle + tolerance_deg):
                return "Downward intent produced a non-downward target angle."

    normalized = _normalize(voice_text)
    if _is_directionless_confirmation(normalized):
        return "Directionless confirmation should be clarified before changing height."

    if _is_directionless_adjustment_request(normalized):
        return "Directionless adjustment request should be clarified before changing height."

    if _is_ambiguous_asr_phrase(normalized):
        return "Ambiguous ASR-like phrase lacks a clear height/posture intent."

    return ""


def _is_directionless_confirmation(normalized: str) -> bool:
    return any(normalized == _normalize(phrase) for phrase in DIRECTIONLESS_CONFIRMATION_PHRASES)


def _is_directionless_modifier(normalized: str) -> bool:
    return any(normalized == _normalize(phrase) for phrase in DIRECTIONLESS_MODIFIER_PHRASES)


def _is_directionless_adjustment_request(normalized: str) -> bool:
    if not _has_any(normalized, DIRECTIONLESS_ADJUSTMENT_HINTS):
        return False
    return not _has_any(normalized, DIRECTION_OR_MAINTAIN_HINTS)


def _is_ambiguous_asr_phrase(normalized: str) -> bool:
    if not _has_any(normalized, AMBIGUOUS_ASR_PHRASES):
        return False
    return not _has_any(normalized, HEIGHT_POSTURE_HINTS)


def _is_tts_echo(text: str) -> bool:
    return _has_any(_normalize(text), IGNORED_TTS_ECHO_PHRASES)


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

        if context == "adjustment_response" and _has_any(normalized, UPWARD_ADJUST_KEYWORDS):
            return "adjust_up"

        if context == "adjustment_response" and _has_any(normalized, DOWNWARD_ADJUST_KEYWORDS):
            return "adjust_down"

        if context == "adjustment_response" and _has_any(normalized, APPROVE_KEYWORDS):
            return "ask_clarification"

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
            response = self._create_completion(system_prompt, user_content)
            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content.strip())
            return self._result_from_json(parsed, raw_text)
        except Exception as exc:
            if "json_validate_failed" in str(exc):
                try:
                    repair_prompt = (
                        system_prompt
                        + "\nCRITICAL JSON REPAIR: Return final numeric literals only. "
                        + "Compute every addition or subtraction before writing JSON. "
                        + "Do not put arithmetic operators in any JSON value."
                    )
                    response = self._create_completion(repair_prompt, user_content)
                    content = response.choices[0].message.content or "{}"
                    parsed = json.loads(content.strip())
                    return self._result_from_json(parsed, raw_text)
                except Exception as retry_exc:
                    exc = retry_exc
            return LlmAdjustmentDecision(
                raw_text=raw_text,
                source="llm_error",
                is_invalid=True,
                reason=f"LLM intent parsing failed: {exc}",
            )

    def _create_completion(self, system_prompt: str, user_content: str):
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=self.temperature,
            max_tokens=200,
            response_format={"type": "json_object"},
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
            direction=str(parsed.get("direction", "unclear")).lower(),
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
