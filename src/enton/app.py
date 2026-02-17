from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from collections import deque
from pathlib import Path

from enton.action.voice import Voice
from enton.cognition.brain import EntonBrain
from enton.cognition.desires import DesireEngine
from enton.cognition.dream import DreamMode
from enton.cognition.fuser import Fuser
from enton.cognition.metacognition import MetaCognitiveEngine
from enton.cognition.persona import REACTION_TEMPLATES, build_system_prompt
from enton.cognition.planner import Planner
from enton.cognition.prediction import PredictionEngine, WorldState
from enton.core.awareness import AwarenessStateMachine
from enton.core.blob_store import BlobStore
from enton.core.commonsense import CommonsenseKB
from enton.core.config import settings
from enton.core.events import (
    ActivityEvent,
    DetectionEvent,
    EmotionEvent,
    EventBus,
    FaceEvent,
    SceneChangeEvent,
    SoundEvent,
    SpeechRequest,
    SystemEvent,
    TranscriptionEvent,
)
from enton.core.knowledge_crawler import KnowledgeCrawler
from enton.core.lifecycle import Lifecycle
from enton.core.memory import Episode, Memory
from enton.core.memory_tiers import MemoryTiers
from enton.core.self_model import SelfModel
from enton.core.visual_memory import VisualMemory
from enton.perception.ears import Ears
from enton.perception.viewer import Viewer
from enton.perception.vision import Vision
from enton.providers.android_bridge import AndroidBridge, find_adb
from enton.skills._shell_state import ShellState
from enton.skills.ai_delegate_toolkit import AIDelegateTools
from enton.skills.android_toolkit import AndroidTools
from enton.skills.blob_toolkit import BlobTools
from enton.skills.describe_toolkit import DescribeTools
from enton.skills.face_toolkit import FaceTools
from enton.skills.file_toolkit import FileTools
from enton.skills.forge_engine import ForgeEngine
from enton.skills.forge_toolkit import ForgeTools
from enton.skills.greet import GreetSkill
from enton.skills.knowledge_toolkit import KnowledgeTools
from enton.skills.memory_toolkit import MemoryTools
from enton.skills.planner_toolkit import PlannerTools
from enton.skills.ptz_toolkit import PTZTools
from enton.skills.react import ReactSkill
from enton.skills.search_toolkit import SearchTools
from enton.skills.shell_toolkit import ShellTools
from enton.skills.skill_registry import SkillRegistry
from enton.skills.system_toolkit import SystemTools
from enton.skills.visual_memory_toolkit import VisualMemoryTools

logger = logging.getLogger(__name__)


class App:
    def __init__(self, viewer: bool = False) -> None:
        self._viewer = viewer
        self._thoughts: deque[str] = deque(maxlen=6)
        self.bus = EventBus()
        self.self_model = SelfModel(settings)
        self.memory = Memory()
        self.blob_store = BlobStore(
            root=settings.blob_store_root,
            fallback=settings.blob_store_fallback,
            qdrant_url=settings.qdrant_url,
        )
        self.vision = Vision(settings, self.bus)
        self.ears = Ears(settings, self.bus, blob_store=self.blob_store)
        self.voice = Voice(settings, ears=self.ears)
        self.fuser = Fuser()
        
        # UI
        self.viewer = Viewer(self.vision, self._thoughts) if viewer else None

        # Phase 10 — Living Entity
        self.desires = DesireEngine()
        self.planner = Planner()
        self.lifecycle = Lifecycle()

        # v0.2.0 — Consciousness
        self.awareness = AwarenessStateMachine()
        self.metacognition = MetaCognitiveEngine()
        self.prediction = PredictionEngine()

        # v0.3.0 — Memory Tiers
        self.visual_memory = VisualMemory(
            qdrant_url=settings.qdrant_url,
            siglip_model=settings.siglip_model,
            siglip_pretrained=settings.siglip_pretrained,
            frames_dir=Path(settings.frames_dir),
            blob_store=self.blob_store,
        )
        self.knowledge_crawler = KnowledgeCrawler(
            qdrant_url=settings.qdrant_url,
        )
        self.commonsense = CommonsenseKB(qdrant_url=settings.qdrant_url)
        self.memory_tiers = MemoryTiers(
            memory=self.memory,
            visual_memory=self.visual_memory,
            knowledge=self.knowledge_crawler,
            commonsense=self.commonsense,
        )

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
            VisualMemoryTools(self.visual_memory),
            KnowledgeTools(self.knowledge_crawler),
            BlobTools(self.blob_store),
        ]

        # Agno-powered Brain with tool calling + fallback chain
        self.brain = EntonBrain(
            settings=settings,
            toolkits=toolkits,
            knowledge=self.memory.knowledge,
        )
        describe_tools._brain = self.brain  # resolve circular dep
        self.knowledge_crawler._brain = self.brain

        # v0.4.0 — Self-Evolution (SkillRegistry + ToolForge)
        self.skill_registry = SkillRegistry(
            brain=self.brain, bus=self.bus, skills_dir=settings.skills_dir,
        )
        self.forge_engine = ForgeEngine(
            brain=self.brain,
            skills_dir=Path(settings.skills_dir),
            sandbox_timeout=settings.forge_sandbox_timeout,
            max_retries=settings.forge_max_retries,
        )
        forge_tools = ForgeTools(
            forge=self.forge_engine, registry=self.skill_registry,
        )
        self.brain.register_toolkit(forge_tools, "_forge_tools")

        # v0.5.0 — AI Delegation (Claude Code + Gemini CLI as tools)
        ai_delegate = AIDelegateTools()
        self.brain.register_toolkit(ai_delegate, "_ai_delegate")

        # v0.6.0 — Android Phone Control (USB + WiFi + 4G via Tailscale)
        if settings.phone_enabled:
            adb_path = find_adb(settings.phone_adb_path)
            if adb_path:
                bridge = AndroidBridge(
                    adb_path=adb_path,
                    device_serial=settings.phone_serial,
                    wifi_host=settings.phone_wifi_host,
                    wifi_port=settings.phone_wifi_port,
                )
                self.brain.register_toolkit(AndroidTools(bridge), "_android_tools")
                logger.info("Android phone control enabled (adb: %s)", adb_path)
            else:
                logger.info("ADB not found — Android phone control disabled")

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
        self.bus.on(SceneChangeEvent, self._on_scene_change)

    def _attach_skills(self) -> None:
        self.greet_skill.attach(self.bus)
        self.react_skill.attach(self.bus)

    async def _on_scene_change(self, event: SceneChangeEvent) -> None:
        """Embed keyframe on significant scene change."""
        cam = self.vision.cameras.get(event.camera_id)
        if cam is None or cam.last_frame is None:
            return
        detections = [d.label for d in (cam.last_detections or [])]
        await self.visual_memory.remember_scene(
            cam.last_frame, detections, event.camera_id,
        )

    async def _on_detection(self, event: DetectionEvent) -> None:
        self.self_model.record_detection(event.label)
        self.memory_tiers.update_object_location(
            event.label, event.camera_id, event.bbox, event.confidence,
        )
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
        tier_ctx = self.memory_tiers.context_string()
        if tier_ctx:
            system += f"\nMEMORY TIERS: {tier_ctx}"
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
                tg.create_task(self.skill_registry.run(), name="skill_registry")
                tg.create_task(self._prediction_loop(), name="prediction")
                if self._sound_detector:
                    tg.create_task(
                        self._sound_detection_loop(), name="sound_detect",
                    )
                if self._metrics:
                    tg.create_task(self._metrics.run(), name="metrics")
                if self.viewer:
                    tg.create_task(self.viewer.run(), name="viewer")
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

    async def _prediction_loop(self) -> None:
        """Active Inference loop: Predict, Compare, Update, Optimize."""
        await asyncio.sleep(5)  # Let sensors warmup
        
        while True:
            await asyncio.sleep(2.0)
            
            # 1. Build WorldState from current sensors
            user_present = self._person_present
            
            # Infer activity level from vision detections/motion
            activity_level = "low"
            if user_present:
                if self.vision.last_activities:
                    # If we have pose data, assume medium/high
                    activity_level = "medium"
                if len(self.vision.last_detections) > 3:
                     activity_level = "high"
            
            state = WorldState(
                timestamp=time.time(),
                user_present=user_present,
                activity_level=activity_level,
            )
            
            # 2. Tick Prediction Engine
            surprise = self.prediction.tick(state)
            
            # 3. Dynamic Optimization (Surprise Minimization)
            if surprise < 0.2:
                # Bored/Expected -> Low FPS
                target_fps = 1.0
                if self.vision.fps > 2.0:  # Using current fps as proxy if target_fps not exposed property, but vision has set_target_fps method
                    # Wait, vision.fps is current actual fps. We want to check target but property is not exposed.
                    # Just set it, it's cheap.
                    pass
                if getattr(self.vision, "_target_fps", 30.0) > 2.0:
                     logger.debug("Boredom (%.2f) -> Low FPS", surprise)
                target_fps = 1.0
            elif surprise > 0.8:
                # Shock/Novelty -> Max FPS
                target_fps = 30.0
                if getattr(self.vision, "_target_fps", 30.0) < 20.0:
                    logger.info("Surprise (%.2f) -> High FPS", surprise)
                    self._push_thought(f"[surpresa] !? ({surprise:.2f})")
            else:
                target_fps = 10.0
                
            self.vision.set_target_fps(target_fps)
            
            # 4. Trigger Investigations (placeholder)

