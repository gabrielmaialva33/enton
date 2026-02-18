"""Android phone control toolkit — gives Enton "sudo" over the phone.

Exposes ADB-powered tools to the LLM for full phone control: shell
commands, screenshots, input events, app management, and file transfer.
All without root on the phone.

Requires: ADB connected (USB or WiFi) to the target device.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agno.tools import Toolkit

if TYPE_CHECKING:
    from enton.providers.android_bridge import AndroidBridge

logger = logging.getLogger(__name__)


class AndroidTools(Toolkit):
    """Controle total do celular Android via ADB — sem root.

    Inspirado em OpenClaw/PhoneClaw: visao de tela via VLM, automacao
    cross-app, monitoramento 24/7 da vida digital do usuario.
    """

    def __init__(self, bridge: AndroidBridge, brain: object | None = None) -> None:
        super().__init__(name="android_tools")
        self._bridge = bridge
        self._brain = brain  # for VLM screen understanding
        self.register(self.phone_status)
        self.register(self.phone_shell)
        self.register(self.phone_screenshot)
        self.register(self.phone_tap)
        self.register(self.phone_swipe)
        self.register(self.phone_type)
        self.register(self.phone_key)
        self.register(self.phone_open_app)
        self.register(self.phone_open_url)
        self.register(self.phone_apps)
        self.register(self.phone_install)
        self.register(self.phone_uninstall)
        self.register(self.phone_push_file)
        self.register(self.phone_pull_file)
        self.register(self.phone_info)
        # v0.6.1 — Life Access + WiFi + VLM Vision
        self.register(self.phone_wifi_setup)
        self.register(self.phone_wifi_connect)
        self.register(self.phone_network_info)
        self.register(self.phone_contacts)
        self.register(self.phone_sms)
        self.register(self.phone_sms_send)
        self.register(self.phone_notifications)
        self.register(self.phone_location)
        self.register(self.phone_call_log)
        self.register(self.phone_calendar)
        self.register(self.phone_see_screen)
        self.register(self.phone_auto_connect)

    # ------------------------------------------------------------------
    # Status & Info
    # ------------------------------------------------------------------

    async def phone_status(self) -> str:
        """Verifica se o celular Android esta conectado via ADB.

        Args:
            (nenhum)
        """
        try:
            connected = await self._bridge.is_connected()
            if connected:
                info = await self._bridge.device_info()
                parts = [f"{k}: {v}" for k, v in info.items()]
                return "Celular conectado.\n" + "\n".join(parts)
            return "Celular NAO conectado via ADB."
        except Exception as e:
            return f"Erro ao verificar celular: {e}"

    async def phone_info(self) -> str:
        """Mostra informacoes detalhadas do celular (modelo, bateria, tela, etc).

        Args:
            (nenhum)
        """
        try:
            info = await self._bridge.device_info()
            if not info:
                return "Celular nao conectado ou nao respondendo."
            lines = [f"  {k}: {v}" for k, v in info.items()]
            return "Info do celular:\n" + "\n".join(lines)
        except Exception as e:
            return f"Erro: {e}"

    # ------------------------------------------------------------------
    # Shell
    # ------------------------------------------------------------------

    async def phone_shell(self, command: str) -> str:
        """Executa um comando shell no celular Android via ADB.

        Args:
            command: Comando shell a executar (ex: 'ls /sdcard', 'whoami', 'cat /proc/cpuinfo').
        """
        try:
            return await self._bridge.shell(command)
        except Exception as e:
            return f"Erro: {e}"

    # ------------------------------------------------------------------
    # Screen
    # ------------------------------------------------------------------

    async def phone_screenshot(self, save_path: str = "/tmp/phone_screenshot.png") -> str:
        """Tira um screenshot do celular e salva como PNG.

        Args:
            save_path: Caminho local pra salvar o PNG (default: /tmp/phone_screenshot.png).
        """
        try:
            png_data = await self._bridge.screenshot()
            if not png_data:
                return "Erro: screenshot vazio — celular conectado?"
            with open(save_path, "wb") as f:
                f.write(png_data)
            return f"Screenshot salvo: {save_path} ({len(png_data)} bytes)"
        except Exception as e:
            return f"Erro: {e}"

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    async def phone_tap(self, x: int, y: int) -> str:
        """Toca na tela do celular nas coordenadas x,y.

        Args:
            x: Coordenada horizontal em pixels.
            y: Coordenada vertical em pixels.
        """
        try:
            return await self._bridge.tap(x, y)
        except Exception as e:
            return f"Erro: {e}"

    async def phone_swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 300,
    ) -> str:
        """Arrasta o dedo na tela do celular de (x1,y1) ate (x2,y2).

        Args:
            x1: X inicial.
            y1: Y inicial.
            x2: X final.
            y2: Y final.
            duration_ms: Duracao do swipe em ms (default: 300).
        """
        try:
            return await self._bridge.swipe(x1, y1, x2, y2, duration_ms)
        except Exception as e:
            return f"Erro: {e}"

    async def phone_type(self, text: str) -> str:
        """Digita texto no campo que esta focado no celular.

        Args:
            text: Texto a digitar.
        """
        try:
            return await self._bridge.input_text(text)
        except Exception as e:
            return f"Erro: {e}"

    async def phone_key(self, key: str) -> str:
        """Pressiona uma tecla no celular (HOME, BACK, ENTER, POWER, VOLUME_UP, etc).

        Args:
            key: Nome da tecla — ex: HOME, BACK, ENTER, POWER, VOLUME_UP, VOLUME_DOWN, CAMERA.
        """
        try:
            return await self._bridge.keyevent(key)
        except Exception as e:
            return f"Erro: {e}"

    # ------------------------------------------------------------------
    # Apps
    # ------------------------------------------------------------------

    async def phone_open_app(self, package: str) -> str:
        """Abre um aplicativo no celular pelo package name.

        Args:
            package: Package name do app (ex: com.whatsapp, com.instagram.android).
        """
        try:
            return await self._bridge.open_app(package)
        except Exception as e:
            return f"Erro: {e}"

    async def phone_open_url(self, url: str) -> str:
        """Abre uma URL no navegador do celular.

        Args:
            url: URL completa (ex: https://google.com).
        """
        try:
            return await self._bridge.open_url(url)
        except Exception as e:
            return f"Erro: {e}"

    async def phone_apps(self, filter_text: str = "") -> str:
        """Lista apps instalados no celular, com filtro opcional.

        Args:
            filter_text: Texto pra filtrar (ex: 'whatsapp', 'google'). Vazio = todos.
        """
        try:
            packages = await self._bridge.list_packages(filter_text)
            if not packages:
                msg = "Nenhum app encontrado"
                if filter_text:
                    msg += f" com filtro '{filter_text}'"
                return msg + "."
            header = f"Apps instalados ({len(packages)}):"
            if filter_text:
                header = f"Apps com '{filter_text}' ({len(packages)}):"
            return header + "\n" + "\n".join(f"  - {p}" for p in packages[:50])
        except Exception as e:
            return f"Erro: {e}"

    async def phone_install(self, apk_path: str) -> str:
        """Instala um APK no celular.

        Args:
            apk_path: Caminho local do arquivo .apk.
        """
        try:
            return await self._bridge.install_apk(apk_path)
        except Exception as e:
            return f"Erro: {e}"

    async def phone_uninstall(self, package: str) -> str:
        """Desinstala um app do celular.

        Args:
            package: Package name do app (ex: com.example.app).
        """
        try:
            return await self._bridge.uninstall(package)
        except Exception as e:
            return f"Erro: {e}"

    # ------------------------------------------------------------------
    # File Transfer
    # ------------------------------------------------------------------

    async def phone_push_file(self, local_path: str, remote_path: str) -> str:
        """Envia um arquivo do PC pro celular.

        Args:
            local_path: Caminho do arquivo no PC.
            remote_path: Caminho destino no celular (ex: /sdcard/Download/arquivo.txt).
        """
        try:
            return await self._bridge.push(local_path, remote_path)
        except Exception as e:
            return f"Erro: {e}"

    async def phone_pull_file(self, remote_path: str, local_path: str) -> str:
        """Puxa um arquivo do celular pro PC.

        Args:
            remote_path: Caminho do arquivo no celular.
            local_path: Caminho destino no PC.
        """
        try:
            return await self._bridge.pull(remote_path, local_path)
        except Exception as e:
            return f"Erro: {e}"

    # ------------------------------------------------------------------
    # WiFi / Network (v0.6.1)
    # ------------------------------------------------------------------

    async def phone_wifi_setup(self) -> str:
        """Ativa ADB via WiFi (precisa estar conectado por USB primeiro).

        Args:
            (nenhum)
        """
        try:
            return await self._bridge.enable_tcpip()
        except Exception as e:
            return f"Erro: {e}"

    async def phone_wifi_connect(self, host: str = "", port: int = 0) -> str:
        """Conecta ao celular via WiFi (TCP/IP deve estar ativo).

        Args:
            host: IP do celular na rede WiFi. Se vazio, usa o ultimo IP detectado.
            port: Porta ADB TCP (default: 5555).
        """
        try:
            return await self._bridge.connect_wifi(host, port)
        except Exception as e:
            return f"Erro: {e}"

    async def phone_network_info(self) -> str:
        """Mostra info de rede do celular (WiFi SSID, IP, velocidade, 4G).

        Args:
            (nenhum)
        """
        try:
            info = await self._bridge.network_info()
            if not info:
                return "Sem info de rede — celular conectado?"
            lines = [f"  {k}: {v}" for k, v in info.items()]
            return "Rede do celular:\n" + "\n".join(lines)
        except Exception as e:
            return f"Erro: {e}"

    # ------------------------------------------------------------------
    # Life Access — Contatos, SMS, Notificacoes, GPS, Calendario
    # ------------------------------------------------------------------

    async def phone_contacts(self, limit: int = 50) -> str:
        """Lista os contatos do celular (nome e numero).

        Args:
            limit: Maximo de contatos a retornar (default: 50).
        """
        try:
            contacts = await self._bridge.contacts(limit)
            if not contacts:
                return "Nenhum contato encontrado."
            lines = [f"  {c['name']}: {c['number']}" for c in contacts]
            return f"Contatos ({len(contacts)}):\n" + "\n".join(lines)
        except Exception as e:
            return f"Erro: {e}"

    async def phone_sms(self, limit: int = 20) -> str:
        """Le as SMS recentes do celular.

        Args:
            limit: Maximo de mensagens a retornar (default: 20).
        """
        try:
            messages = await self._bridge.sms_list(limit)
            if not messages:
                return "Nenhuma SMS encontrada (pode precisar de permissao)."
            lines = []
            for m in messages:
                arrow = "<-" if m["dir"] == "recebido" else "->"
                lines.append(f"  {arrow} {m['from']}: {m['text'][:80]}")
            return f"SMS recentes ({len(messages)}):\n" + "\n".join(lines)
        except Exception as e:
            return f"Erro: {e}"

    async def phone_sms_send(self, number: str, message: str) -> str:
        """Abre o app de SMS com numero e mensagem pre-preenchidos.

        Args:
            number: Numero do destinatario (ex: +5511999999999).
            message: Texto da mensagem.
        """
        try:
            return await self._bridge.send_sms(number, message)
        except Exception as e:
            return f"Erro: {e}"

    async def phone_notifications(self, limit: int = 15) -> str:
        """Lista as notificacoes atuais na barra do celular.

        Args:
            limit: Maximo de notificacoes a retornar (default: 15).
        """
        try:
            notifs = await self._bridge.notifications(limit)
            if not notifs:
                return "Nenhuma notificacao no momento."
            lines = []
            for n in notifs:
                lines.append(f"  [{n['app']}] {n['title']}: {n['text'][:60]}")
            return f"Notificacoes ({len(notifs)}):\n" + "\n".join(lines)
        except Exception as e:
            return f"Erro: {e}"

    async def phone_location(self) -> str:
        """Pega a localizacao GPS/rede do celular (ultima conhecida).

        Args:
            (nenhum)
        """
        try:
            loc = await self._bridge.location()
            if not loc:
                return "Sem dados de localizacao."
            lines = [f"  {k}: {v}" for k, v in loc.items()]
            return "Localizacao:\n" + "\n".join(lines)
        except Exception as e:
            return f"Erro: {e}"

    async def phone_call_log(self, limit: int = 20) -> str:
        """Lista as chamadas recentes do celular.

        Args:
            limit: Maximo de chamadas a retornar (default: 20).
        """
        try:
            calls = await self._bridge.call_log(limit)
            if not calls:
                return "Nenhuma chamada no historico (pode precisar de permissao)."
            lines = []
            for c in calls:
                lines.append(f"  {c['type']}: {c['name']} ({c['number']}) — {c['duration']}")
            return f"Chamadas recentes ({len(calls)}):\n" + "\n".join(lines)
        except Exception as e:
            return f"Erro: {e}"

    async def phone_calendar(self, limit: int = 20) -> str:
        """Lista eventos do calendario do celular.

        Args:
            limit: Maximo de eventos a retornar (default: 20).
        """
        try:
            events = await self._bridge.calendar_events(limit)
            if not events:
                return "Nenhum evento no calendario."
            lines = []
            for e in events:
                loc = f" @ {e['location']}" if e.get("location") else ""
                lines.append(f"  - {e['title']}{loc}")
            return f"Eventos ({len(events)}):\n" + "\n".join(lines)
        except Exception as e:
            return f"Erro: {e}"

    # ------------------------------------------------------------------
    # VLM Vision + Auto-connect (OpenClaw-inspired)
    # ------------------------------------------------------------------

    async def phone_see_screen(self, question: str = "") -> str:
        """Tira screenshot do celular e usa VLM pra entender o que esta na tela.

        Estilo OpenClaw/PhoneClaw: visao de tela com AI. Pode responder
        perguntas sobre o que esta exibido, identificar apps abertos,
        ler textos, e sugerir proximas acoes.

        Args:
            question: Pergunta sobre a tela (ex: 'que app esta aberto?'). Se vazio, descreve a tela.
        """
        try:
            png_data = await self._bridge.screenshot()
            if not png_data:
                return "Erro: screenshot vazio — celular conectado?"

            # Try VLM if brain is available
            if self._brain and hasattr(self._brain, "describe_scene"):
                prompt = question or "Descreva o que esta na tela do celular."
                desc = await self._brain.describe_scene(
                    png_data,
                    system=(
                        "Voce esta olhando a tela de um celular Android. "
                        "Descreva o que ve: app aberto, conteudo na tela, "
                        "botoes visiveis, notificacoes. Seja breve e util."
                    ),
                    prompt=prompt,
                )
                if desc:
                    return f"[VLM] {desc}"

            # Fallback: just report screenshot size + basic info
            return (
                f"Screenshot capturado ({len(png_data):,} bytes). "
                "VLM nao disponivel para analise visual."
            )
        except Exception as e:
            return f"Erro: {e}"

    async def phone_auto_connect(self) -> str:
        """Tenta conectar ao celular automaticamente (USB -> WiFi -> Tailscale).

        Args:
            (nenhum)
        """
        try:
            return await self._bridge.auto_connect()
        except Exception as e:
            return f"Erro: {e}"
