from __future__ import annotations

import asyncio
import logging
import math
import random
import re
import time
from collections import deque

import numpy as np  # noqa: TC002 — used at runtime

from enton.action.voice import Voice
from enton.cognition.brain import EntonBrain
from enton.cognition.desires import DesireEngine
from enton.cognition.dream import DreamMode
from enton.cognition.fuser import Fuser
from enton.cognition.metacognition import MetaCognitiveEngine
from enton.cognition.persona import REACTION_TEMPLATES, build_system_prompt
from enton.cognition.planner import Planner
from enton.core.awareness import AwarenessStateMachine
from enton.core.config import settings
from enton.core.events import (
    ActivityEvent,
    DetectionEvent,
    EmotionEvent,
    EventBus,
    FaceEvent,
    SoundEvent,
    SpeechRequest,
    SystemEvent,
    TranscriptionEvent,
)
from enton.core.lifecycle import Lifecycle
from enton.core.memory import Episode, Memory
from enton.core.self_model import SelfModel
from enton.perception.ears import Ears
from enton.perception.vision import Vision
from enton.skills._shell_state import ShellState
from enton.skills.describe_toolkit import DescribeTools
from enton.skills.face_toolkit import FaceTools
from enton.skills.file_toolkit import FileTools
from enton.skills.greet import GreetSkill
from enton.skills.memory_toolkit import MemoryTools
from enton.skills.planner_toolkit import PlannerTools
from enton.skills.ptz_toolkit import PTZTools
from enton.skills.react import ReactSkill
from enton.skills.search_toolkit import SearchTools
from enton.skills.shell_toolkit import ShellTools
from enton.skills.system_toolkit import SystemTools

logger = logging.getLogger(__name__)


class App:
    def __init__(self, viewer: bool = False) -> None:
        self._viewer = viewer
        self._thoughts: deque[str] = deque(maxlen=6)
        self.bus = EventBus()
        self.self_model = SelfModel(settings)
        self.memory = Memory()
        self.vision = Vision(settings, self.bus)
        self.ears = Ears(settings, self.bus)
        self.voice = Voice(settings, ears=self.ears)
        self.fuser = Fuser()

        # Phase 10 — Living Entity
        self.desires = DesireEngine()
        self.planner = Planner()
        self.lifecycle = Lifecycle()

        # v0.2.0 — Consciousness
        self.awareness = AwarenessStateMachine()
        self.metacognition = MetaCognitiveEngine()

        # Agno Toolkits
        shell_state = ShellState()
        describe_tools = DescribeTools(self.vision)
        toolkits = [
            describe_tools,
            FaceTools(self.vision, self.vision.face_recognizer),
            FileTools(shell_state),
            MemoryTools(self.memory),
            PlannerTools(self.planner),
            PTZTools(),
            SearchTools(),
            ShellTools(shell_state),
            SystemTools(),
        ]

        # Agno-powered Brain with tool calling + fallback chain
        self.brain = EntonBrain(
            settings=settings,
            toolkits=toolkits,
            knowledge=self.memory.knowledge,
        )
        describe_tools._brain = self.brain  # resolve circular dep

        # Dream mode (must be after brain + memory)
        self.dream = DreamMode(memory=self.memory, brain=self.brain)

        # Event-driven skills (not Agno tools — react to EventBus)
        self.greet_skill = GreetSkill(self.voice, self.memory)
        self.react_skill = ReactSkill(self.voice, self.memory)

        self._person_present: bool = False
        self._last_person_seen: float = 0
        self._sound_detector = None
        self._metrics = None
        self._init_sound_detector()
        self._init_metrics()
        self._register_handlers()
        self._attach_skills()
        self._probe_capabilities()

    def _init_metrics(self) -> None:
        try:
            from enton.core.metrics import MetricsCollector

            self._metrics = MetricsCollector(
                dsn=settings.timescale_dsn,
                interval=settings.metrics_interval,
            )
            self._metrics.register("engagement", lambda: self.self_model.mood.engagement)
            self._metrics.register("social", lambda: self.self_model.mood.social)
            self._metrics.register("vision_fps", lambda: self.vision.fps)
            logger.info("MetricsCollector initialized")
        except Exception:
            logger.warning("MetricsCollector unavailable")

    def _init_sound_detector(self) -> None:
        try:
            from enton.perception.sounds import SoundDetector

            self._sound_detector = SoundDetector(threshold=0.3)
            logger.info("SoundDetector initialized")
        except Exception:
            logger.warning("SoundDetector unavailable")

    def _push_thought(self, text: str) -> None:
        """Add a thought to the viewer display buffer."""
        # Truncate long thoughts for the HUD
        if len(text) > 120:
            text = text[:117] + "..."
        self._thoughts.append(text)

    def _probe_capabilities(self) -> None:
        sm = self.self_model.senses
        sm.llm_ready = bool(self.brain._models)
        if self.brain._models:
            mid = getattr(self.brain._models[0], "id", "unknown")
            sm.active_providers["llm"] = mid
        sm.tts_ready = bool(self.voice._providers)
        if self.voice._providers:
            sm.active_providers["tts"] = str(self.voice._primary)
        sm.stt_ready = bool(self.ears._providers)
        if self.ears._providers:
            sm.active_providers["stt"] = str(self.ears._primary)

    def _register_handlers(self) -> None:
        self.bus.on(DetectionEvent, self._on_detection)
        self.bus.on(ActivityEvent, self._on_activity)
        self.bus.on(EmotionEvent, self._on_emotion)
        self.bus.on(TranscriptionEvent, self._on_transcription)
        self.bus.on(FaceEvent, self._on_face)
        self.bus.on(SoundEvent, self._on_sound)
        self.bus.on(SpeechRequest, self._on_speech_request)
        self.bus.on(SystemEvent, self._on_system_event)

    def _attach_skills(self) -> None:
        self.greet_skill.attach(self.bus)
        self.react_skill.attach(self.bus)

    async def _on_detection(self, event: DetectionEvent) -> None:
        self.self_model.record_detection(event.label)
        if event.label == "person":
            self._person_present = True
            self._last_person_seen = time.time()

    async def _on_activity(self, event: ActivityEvent) -> None:
        self.self_model.record_activity(event.activity)

    async def _on_emotion(self, event: EmotionEvent) -> None:
        self.self_model.record_emotion(event.emotion)

    async def _on_face(self, event: FaceEvent) -> None:
        if event.identity != "unknown":
            self._push_thought(f"[face] {event.identity} ({event.confidence:.0%})")
            logger.info(
                "Face recognized: %s (%.0f%%)",
                event.identity, event.confidence * 100,
            )
            self.memory.learn_about_user(
                f"Rosto reconhecido: {event.identity}",
            )
            # Greet recognized person (with cooldown via react_skill)
            if not self.voice.is_speaking:
                template = random.choice(REACTION_TEMPLATES["face_recognized"])
                await self.voice.say(template.format(name=event.identity))

    async def _on_sound(self, event: SoundEvent) -> None:
        logger.info("Sound: %s (%.0f%%)", event.label, event.confidence * 100)
        self.self_model.record_sound(event.label, event.confidence)
        self._push_thought(f"[som] {event.label} ({event.confidence:.0%})")
        self.desires.on_sound(event.label)

        if self.voice.is_speaking:
            return

        # High-priority sounds get instant reactions (no brain call)
        urgent_reactions = {
            "Alarme": "Eita, alarme! Tá tudo bem?",
            "Sirene": "Sirene! O que tá acontecendo?",
            "Vidro quebrando": "Caramba, que barulho foi esse?!",
        }
        reaction = urgent_reactions.get(event.label)
        if reaction:
            self.awareness.trigger_alert(f"sound:{event.label}", self.bus)
            await self.voice.say(reaction)
            return

        # Other sounds: ask brain for intelligent reaction
        if event.confidence > 0.5:
            prompt = (
                f"Acabei de ouvir um som ambiente: '{event.label}' "
                f"(confianca {event.confidence:.0%}). "
                "Faca um comentario curto e natural sobre isso em 1 frase."
            )
            response = await self.brain.think(
                prompt,
                system="Voce e o Enton. Comente brevemente sobre o som detectado.",
            )
            if response and not self.voice.is_speaking:
                await self.voice.say(response)

    async def _on_transcription(self, event: TranscriptionEvent) -> None:
        if not event.text.strip():
            return

        # Partial transcription — show in viewer but don't process
        if not event.is_final:
            self._push_thought(f"[ouvindo] {event.text[:80]}...")
            return

        self.self_model.record_interaction()
        self.memory.strengthen_relationship()
        self.desires.on_interaction()
        self.dream.on_interaction()
        self.awareness.on_interaction(self.bus)

        # Extract basic facts (simple heuristic for now)
        self._extract_facts(event.text)

        # Build context using Fuser with all available perception data
        detections = self.vision.last_detections
        activities = self.vision.last_activities
        emotions = self.vision.last_emotions
        scene_desc = self.fuser.fuse(detections, activities, emotions)
        
        system = build_system_prompt(
            self.self_model,
            self.memory,
            detections=[{"label": d.label} for d in detections],
        )
        
        # Inject Fuser context into system prompt or user message
        # Let's prepend to the user message or append to system
        system += f"\n\nCONTEXTO VISUAL ATUAL: {scene_desc}"
        system += f"\nAWARENESS: {self.awareness.summary()}"
        system += f"\nMETACOGNITION: {self.metacognition.introspect()}"
        system += "\nVocê tem acesso a ferramentas. Use-as se necessário para responder."

        self._push_thought(f"[ouviu] {event.text[:80]}")

        # Metacognitive-wrapped brain call
        trace = self.metacognition.begin_trace(event.text, strategy="agent")
        response = await self.brain.think_agent(event.text, system=system)
        provider = getattr(self.brain._agent.model, "id", "?")
        self.metacognition.end_trace(
            trace, response or "",
            provider=provider, success=bool(response),
        )

        if response:
            self._push_thought(f"[brain] {response[:100]}")
            await self.voice.say(response)
            self.memory.remember(
                Episode(
                    kind="conversation",
                    summary=f"User: '{event.text[:60]}' -> Me: '{response[:60]}'",
                    tags=["chat"],
                )
            )

    def _extract_facts(self, text: str) -> None:
        # Simple regex extraction for Phase 1
        patterns = [
            (r"(?:meu nome é|eu sou o|me chamo) (.+)", "name"),
            (r"(?:eu gosto de|adoro|amo) (.+)", "like"),
        ]
        text_lower = text.lower()
        for pattern, kind in patterns:
            match = re.search(pattern, text_lower)
            if match:
                fact = match.group(1).strip()
                if kind == "name":
                    self.memory.learn_about_user(f"Nome é {fact.title()}")
                else:
                    self.memory.learn_about_user(f"Gosta de {fact}")

    async def _on_speech_request(self, event: SpeechRequest) -> None:
        await self.voice.say(event.text)

    async def _on_system_event(self, event: SystemEvent) -> None:
        if event.kind == "startup":
            text = random.choice(REACTION_TEMPLATES["startup"])
            await self.voice.say(text)
            self.memory.remember(
                Episode(
                    kind="system",
                    summary="Enton booted up",
                    tags=["startup"],
                )
            )
        elif event.kind == "camera_lost":
            self.self_model.senses.camera_online = False
            logger.warning("Camera connection lost")
        elif event.kind == "camera_connected":
            self.self_model.senses.camera_online = True

    async def run(self) -> None:
        logger.info("Enton starting up...")

        # Lifecycle — restore state from previous session
        wake_msg = self.lifecycle.on_boot(self.self_model, self.desires)
        logger.info("Lifecycle: %s", self.lifecycle.summary())
        logger.info("Self-state: %s", self.self_model.introspect())

        await self.bus.emit(SystemEvent(kind="startup"))
        if wake_msg:
            await self.voice.say(wake_msg)

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self.bus.run(), name="event_bus")
                tg.create_task(self.vision.run(), name="vision")
                tg.create_task(self.voice.run(), name="voice")
                tg.create_task(self.ears.run(), name="ears")
                tg.create_task(self._idle_loop(), name="idle")
                tg.create_task(self._mood_decay_loop(), name="mood_decay")
                tg.create_task(self._scene_description_loop(), name="scene_desc")
                tg.create_task(self._desire_loop(), name="desires")
                tg.create_task(self._planner_loop(), name="planner")
                tg.create_task(self._autosave_loop(), name="autosave")
                tg.create_task(self._awareness_loop(), name="awareness")
                tg.create_task(self.dream.run(), name="dream")
                if self._sound_detector:
                    tg.create_task(
                        self._sound_detection_loop(), name="sound_detect",
                    )
                if self._metrics:
                    tg.create_task(self._metrics.run(), name="metrics")
                if self._viewer:
                    tg.create_task(self._viewer_loop(), name="viewer")
        finally:
            # Graceful shutdown — persist state
            self.lifecycle.on_shutdown(self.self_model, self.desires)
            logger.info("Enton shutdown. State saved.")

    async def _idle_loop(self) -> None:
        idle_tick = 0
        while True:
            await asyncio.sleep(1.0)
            idle_tick += 1

            now = time.time()
            # Person left logic
            if self._person_present and (now - self._last_person_seen > settings.idle_timeout):
                self._person_present = False
                self.greet_skill.reset_presence()
                if self.self_model.mood.engagement > 0.3:
                    await self.voice.say("Opa, até mais!")
                await self.bus.emit(SystemEvent(kind="person_left"))

            # Decay engagement slowly (every 30s, not every 1s)
            if idle_tick % 30 == 0:
                self.self_model.mood.on_idle()

    async def _mood_decay_loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            self.self_model.mood.tick()

    async def _scene_description_loop(self) -> None:
        """Periodically describes the scene using VLM if a person is present."""
        while True:
            await asyncio.sleep(settings.scene_describe_interval)

            if not self._person_present:
                continue

            if self.self_model.mood.engagement < 0.4:
                continue

            if self.voice.is_speaking:
                continue

            # Try VLM with actual camera frame
            jpeg = self.vision.get_frame_jpeg()
            if jpeg is not None:
                response = await self.brain.describe_scene(
                    jpeg,
                    system=(
                        "Você é o Enton, um robô assistente zoeiro. "
                        "Comente algo breve e interessante sobre a cena."
                    ),
                )
                if response:
                    await self.voice.say(response)
                    continue

            # Fallback: Fuser text-only if no VLM available
            detections = self.vision.last_detections
            if not detections:
                continue
            activities = self.vision.last_activities
            emotions = self.vision.last_emotions
            scene_desc = self.fuser.fuse(detections, activities, emotions)
            if "Nenhum objeto" in scene_desc:
                continue
            prompt = (
                "Comente algo breve e interessante sobre esta cena. "
                f"Contexto: {scene_desc}"
            )
            response = await self.brain.think(
                prompt, system="Você é o Enton, um robô observador curioso.",
            )
            if response:
                await self.voice.say(response)

    async def _sound_detection_loop(self) -> None:
        """Captures audio chunks and classifies ambient sounds."""
        import sounddevice as sd

        sample_rate = 48000
        chunk_duration = 2.0  # seconds
        chunk_samples = int(sample_rate * chunk_duration)
        cooldown = 10.0  # seconds between sound reactions
        last_reaction = 0.0

        logger.info("Sound detection loop started (sr=%d)", sample_rate)

        while True:
            try:
                # Skip if ears are actively listening to speech
                if self.ears.muted:
                    await asyncio.sleep(chunk_duration)
                    continue

                loop = asyncio.get_running_loop()

                def _record():
                    return sd.rec(
                        chunk_samples,
                        samplerate=sample_rate,
                        channels=1,
                        dtype="float32",
                    )

                audio = await loop.run_in_executor(None, _record)
                await asyncio.sleep(chunk_duration)
                sd.wait()

                audio = audio.squeeze()
                if audio.max() < 0.01:
                    continue  # silence

                now = time.time()
                if now - last_reaction < cooldown:
                    continue

                results = await self._sound_detector.classify_async(
                    audio, sample_rate,
                )
                for r in results:
                    logger.info(
                        "Sound event: %s (%.0f%%)",
                        r.label, r.confidence * 100,
                    )
                    await self.bus.emit(
                        SoundEvent(
                            label=r.label,
                            confidence=r.confidence,
                        )
                    )
                    last_reaction = now
                    break  # only react to top result

            except Exception:
                logger.exception("Sound detection error")
                await asyncio.sleep(5.0)

    async def _desire_loop(self) -> None:
        """Autonomous desire engine — Enton acts on his own wants."""
        await asyncio.sleep(30)  # Let everything initialize first

        while True:
            await asyncio.sleep(10)

            # Tick desires based on current mood
            self.desires.tick(self.self_model, dt=10)

            # Check if any desire should activate
            desire = self.desires.get_active_desire()
            if desire is None:
                continue

            if self.voice.is_speaking:
                continue

            self._push_thought(f"[desejo] {desire.name} (urgencia={desire.urgency:.1f})")
            logger.info("Desire activated: %s (urgency=%.2f)", desire.name, desire.urgency)
            desire.activate()

            # Act on the desire
            if desire.name == "socialize":
                prompt = self.desires.get_prompt(desire)
                await self.voice.say(prompt)

            elif desire.name == "observe":
                self.desires.on_observation()
                jpeg = self.vision.get_frame_jpeg()
                if jpeg is not None:
                    desc = await self.brain.describe_scene(
                        jpeg,
                        system="Você é o Enton. Comente algo curto sobre a cena.",
                    )
                    if desc:
                        await self.voice.say(desc)

            elif desire.name == "learn":
                # Use brain with search tool to learn something
                response = await self.brain.think_agent(
                    "Pesquise algo interessante e curioso e me conte em 1-2 frases.",
                    system="Você é o Enton, curioso sobre o mundo.",
                )
                if response:
                    await self.voice.say(response)

            elif desire.name == "check_on_user":
                prompt = self.desires.get_prompt(desire)
                await self.voice.say(prompt)

            elif desire.name == "optimize":
                response = await self.brain.think_agent(
                    "Verifique o status do sistema (CPU, RAM, GPU) e me diga se está tudo ok.",
                )
                if response:
                    await self.voice.say(response)

            elif desire.name == "reminisce":
                episodes = self.memory.recent(3)
                if episodes:
                    ep = random.choice(episodes)
                    await self.voice.say(f"Lembrei... {ep.summary}")

            elif desire.name == "create":
                self.desires.on_creation()
                response = await self.brain.think_agent(
                    "Crie algo curto e criativo: um haiku, piada nerdy, "
                    "ou dica de programacao. Escolha aleatoriamente.",
                    system="Voce e o Enton, criativo e zoeiro.",
                )
                if response:
                    await self.voice.say(response)

            elif desire.name == "explore":
                # Try to move camera via PTZ, then describe
                response = await self.brain.think_agent(
                    "Mova a camera para uma direcao aleatoria "
                    "e descreva o que voce ve.",
                    system="Voce e o Enton. Use as ferramentas PTZ e describe.",
                )
                if response:
                    await self.voice.say(response)

            elif desire.name == "play":
                response = await self.brain.think_agent(
                    "Conte uma piada curta, um fato curioso, "
                    "ou proponha um quiz rapido pro Gabriel.",
                    system="Voce e o Enton, zoeiro. Seja divertido e breve.",
                )
                if response:
                    await self.voice.say(response)

    async def _planner_loop(self) -> None:
        """Checks for due reminders and routines."""
        await asyncio.sleep(10)

        while True:
            await asyncio.sleep(30)

            # Check reminders
            due = self.planner.get_due_reminders()
            for r in due:
                logger.info("Reminder due: %s", r.text)
                if not self.voice.is_speaking:
                    await self.voice.say(f"Lembrete: {r.text}")

            # Check routines
            import datetime

            hour = datetime.datetime.now().hour
            routines = self.planner.get_due_routines(hour)
            for routine in routines:
                logger.info("Routine due: %s", routine["name"])
                if not self.voice.is_speaking:
                    await self.voice.say(routine["text"])

    async def _viewer_loop(self) -> None:
        """Live vision window — cv2-only HUD with multi-camera grid support."""
        import cv2

        fullscreen = False
        grid_mode = len(self.vision.cameras) > 1
        scan_y = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_sm = cv2.FONT_HERSHEY_PLAIN

        cv2.namedWindow("Enton Vision", cv2.WINDOW_NORMAL)
        cam_ids = list(self.vision.cameras.keys())
        logger.info(
            "Viewer opened (%d camera%s)",
            len(cam_ids), "s" if len(cam_ids) != 1 else "",
        )

        while True:
            if grid_mode:
                annotated = self._build_grid(cam_ids, font, font_sm)
            else:
                annotated = self._annotate_camera(
                    self.vision.cameras[cam_ids[0]], font, font_sm,
                )

            if annotated is None:
                await asyncio.sleep(0.1)
                continue

            fh, fw = annotated.shape[:2]

            # Scan line
            scan_y = (scan_y + 3) % fh
            cv2.line(annotated, (0, scan_y), (fw, scan_y), (0, 255, 120), 1)

            # Thoughts panel (bottom)
            if self._thoughts:
                y_base = fh - 30
                for thought in reversed(self._thoughts):
                    pt = (12, y_base)
                    cv2.putText(annotated, thought, pt, font, 0.45, (0, 0, 0), 3)
                    cv2.putText(annotated, thought, pt, font, 0.45, (0, 255, 200), 1)
                    y_base -= 20

            cv2.imshow("Enton Vision", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                raise KeyboardInterrupt
            elif key == ord("f"):
                fullscreen = not fullscreen
                prop = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty("Enton Vision", cv2.WND_PROP_FULLSCREEN, prop)
            elif key == ord("g"):
                grid_mode = not grid_mode
            elif key == ord("c"):
                # Cycle active camera
                idx = cam_ids.index(self.vision.active_camera_id)
                nxt = cam_ids[(idx + 1) % len(cam_ids)]
                self.vision.switch_camera(nxt)
                self._push_thought(f"[cam] -> {nxt}")
            elif ord("1") <= key <= ord("9"):
                idx = key - ord("1")
                if idx < len(cam_ids):
                    self.vision.switch_camera(cam_ids[idx])

            await asyncio.sleep(0.01)

    def _annotate_camera(self, cam, font, font_sm) -> np.ndarray | None:
        """Draw HUD overlay on a single camera frame."""
        import cv2

        frame = cam.last_frame
        if frame is None:
            return None

        annotated = frame.copy()
        fh, fw = annotated.shape[:2]

        # Detection overlays
        for det in cam.last_detections:
            if det.bbox:
                color = (0, 255, 120) if det.label == "person" else (255, 160, 0)
                if det.label in ("cat", "dog"):
                    color = (0, 200, 255)
                x1, y1, x2, y2 = det.bbox
                bw, bh = x2 - x1, y2 - y1
                c = max(15, min(bw, bh) // 5)
                cv2.line(annotated, (x1, y1), (x1 + c, y1), color, 2)
                cv2.line(annotated, (x1, y1), (x1, y1 + c), color, 2)
                cv2.line(annotated, (x2, y1), (x2 - c, y1), color, 2)
                cv2.line(annotated, (x2, y1), (x2, y1 + c), color, 2)
                cv2.line(annotated, (x1, y2), (x1 + c, y2), color, 2)
                cv2.line(annotated, (x1, y2), (x1, y2 - c), color, 2)
                cv2.line(annotated, (x2, y2), (x2 - c, y2), color, 2)
                cv2.line(annotated, (x2, y2), (x2, y2 - c), color, 2)
                lbl = f"{det.label} {det.confidence:.0%}"
                pt = (x1, y1 - 6)
                cv2.putText(annotated, lbl, pt, font_sm, 1.0, (0, 0, 0), 3)
                cv2.putText(annotated, lbl, pt, font_sm, 1.0, color, 1)

        # Emotion labels
        for emo in cam.last_emotions:
            if emo.bbox and emo.bbox != (0, 0, 0, 0):
                x1, _, x2, y2 = emo.bbox
                lbl = f"{emo.emotion} {emo.score:.0%}"
                cx = (x1 + x2) // 2
                pt = (cx - 40, y2 + 16)
                cv2.putText(annotated, lbl, pt, font_sm, 1.0, (0, 0, 0), 3)
                cv2.putText(annotated, lbl, pt, font_sm, 1.0, emo.color, 1)

        # HUD panel
        n_persons = sum(1 for d in cam.last_detections if d.label == "person")
        n_obj = len(cam.last_detections) - n_persons
        overlay = annotated.copy()
        cv2.rectangle(overlay, (8, 8), (260, 80), (10, 12, 18), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
        cv2.rectangle(annotated, (8, 8), (260, 80), (0, 255, 120), 1)
        title = f"ENTON [{cam.id}]" if len(self.vision.cameras) > 1 else "ENTON"
        cv2.putText(annotated, title, (16, 34), font, 0.6, (0, 255, 120), 2)
        fps_txt = f"{cam.fps:.0f} fps"
        cv2.putText(annotated, fps_txt, (200, 34), font_sm, 1.0, (80, 90, 100), 1)
        if n_persons:
            status = f"{n_persons} pessoa{'s' if n_persons != 1 else ''}"
        else:
            status = "scanning..."
        if n_obj > 0:
            status += f"  {n_obj} obj"
        cv2.putText(annotated, status, (16, 58), font, 0.45, (0, 210, 230), 1)
        for i, act in enumerate(cam.last_activities[:2]):
            pt = (16, 74 + i * 14)
            cv2.putText(annotated, act.activity, pt, font_sm, 0.9, act.color, 1)

        return annotated

    def _build_grid(self, cam_ids: list[str], font, font_sm) -> np.ndarray | None:
        """Build grid view from multiple cameras."""
        import cv2

        n = len(cam_ids)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        # Target tile size
        tile_w, tile_h = 640, 480
        grid = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

        for idx, cam_id in enumerate(cam_ids):
            cam = self.vision.cameras.get(cam_id)
            if cam is None:
                continue
            annotated = self._annotate_camera(cam, font, font_sm)
            if annotated is None:
                continue
            tile = cv2.resize(annotated, (tile_w, tile_h))
            r, c = divmod(idx, cols)
            y0, x0 = r * tile_h, c * tile_w
            grid[y0:y0 + tile_h, x0:x0 + tile_w] = tile

        return grid if grid.any() else None

    async def _awareness_loop(self) -> None:
        """Evaluate awareness state transitions periodically."""
        await asyncio.sleep(10)
        while True:
            await asyncio.sleep(5)
            self.awareness.evaluate(self.self_model, self.bus)

    async def _autosave_loop(self) -> None:
        """Periodically saves state for crash recovery."""
        while True:
            await asyncio.sleep(300)  # every 5 min
            self.lifecycle.save_periodic(self.self_model, self.desires)
            logger.debug("Autosave complete")
