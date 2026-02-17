"""Agno Toolkit for camera PTZ control (digital ONVIF + physical motor)."""

from __future__ import annotations

import asyncio
import logging

from agno.tools import Toolkit

logger = logging.getLogger(__name__)

# Camera ONVIF endpoint for digital PTZ
ONVIF_HOST = "192.168.18.222"
ONVIF_PORT = 5000

# Motor ioctl via telnet (physical PTZ)
MOTOR_HOST = "192.168.18.222"
MOTOR_TELNET_PORT = 23

# Direction -> (pan, tilt) mapping
_DIRECTION_MAP: dict[str, tuple[float, float]] = {
    "left": (-0.5, 0.0),
    "right": (0.5, 0.0),
    "up": (0.0, 0.5),
    "down": (0.0, -0.5),
    "esquerda": (-0.5, 0.0),
    "direita": (0.5, 0.0),
    "cima": (0.0, 0.5),
    "baixo": (0.0, -0.5),
}


class PTZTools(Toolkit):
    """Camera PTZ control via ONVIF (digital) and motor ioctl (physical)."""

    def __init__(
        self,
        onvif_host: str = ONVIF_HOST,
        onvif_port: int = ONVIF_PORT,
        motor_host: str = MOTOR_HOST,
        motor_telnet_port: int = MOTOR_TELNET_PORT,
    ):
        super().__init__(name="ptz_tools")
        self.onvif_host = onvif_host
        self.onvif_port = onvif_port
        self.motor_host = motor_host
        self.motor_telnet_port = motor_telnet_port
        self.register(self.camera_move)
        self.register(self.camera_motor_move)

    async def _onvif_continuous_move(
        self, pan: float, tilt: float, duration: float = 1.0
    ) -> str:
        """Send ONVIF ContinuousMove then Stop after duration."""
        soap_move = f"""<?xml version="1.0" encoding="utf-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl"
            xmlns:tt="http://www.onvif.org/ver10/schema">
  <s:Body>
    <tptz:ContinuousMove>
      <tptz:ProfileToken>profile_0</tptz:ProfileToken>
      <tptz:Velocity>
        <tt:PanTilt x="{pan}" y="{tilt}"/>
      </tptz:Velocity>
    </tptz:ContinuousMove>
  </s:Body>
</s:Envelope>"""

        soap_stop = """<?xml version="1.0" encoding="utf-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl">
  <s:Body>
    <tptz:Stop>
      <tptz:ProfileToken>profile_0</tptz:ProfileToken>
      <tptz:PanTilt>true</tptz:PanTilt>
      <tptz:Zoom>true</tptz:Zoom>
    </tptz:Stop>
  </s:Body>
</s:Envelope>"""

        url = f"http://{self.onvif_host}:{self.onvif_port}/onvif/ptz_service"
        content_type = "application/soap+xml; charset=utf-8"

        try:
            # Send move command
            proc = await asyncio.create_subprocess_exec(
                "curl", "-s", "-X", "POST", url,
                "-H", f"Content-Type: {content_type}",
                "-d", soap_move,
                "--connect-timeout", "3",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=5.0)

            # Wait for the move duration
            await asyncio.sleep(duration)

            # Send stop command
            proc = await asyncio.create_subprocess_exec(
                "curl", "-s", "-X", "POST", url,
                "-H", f"Content-Type: {content_type}",
                "-d", soap_stop,
                "--connect-timeout", "3",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=5.0)

            return "OK"
        except Exception as e:
            return f"Erro ONVIF: {e}"

    async def camera_move(self, direction: str, duration: float = 1.0) -> str:
        """Move a camera na direcao especificada usando PTZ digital (ONVIF).

        Envia ContinuousMove SOAP para o endpoint ONVIF da camera,
        aguarda a duracao, e envia Stop. Util para ajustar enquadramento.

        Args:
            direction: Direcao: up, down, left, right, stop.
            duration: Duracao do movimento em segundos (default 1.0).
        """
        direction = direction.lower().strip()

        if direction in ("stop", "parar"):
            result = await self._onvif_continuous_move(0, 0, 0)
            return f"Camera parada. {result}"

        if direction not in _DIRECTION_MAP:
            return (
                f"Direcao '{direction}' invalida. "
                "Use: up/down/left/right (ou cima/baixo/esquerda/direita)."
            )

        pan, tilt = _DIRECTION_MAP[direction]
        result = await self._onvif_continuous_move(pan, tilt, duration)
        return f"Camera movida para {direction} por {duration}s. {result}"

    async def camera_motor_move(self, h_steps: int = 0, v_steps: int = 0) -> str:
        """Move o motor fisico da camera (PTZ mecanico) via ioctl remoto.

        O driver multiplica steps por 10x internamente.
        Steps positivos = horario/cima, negativos = anti-horario/baixo.
        O binario motor_mini deve estar presente na camera em /tmp/m ou /mnt/sdcard/motor_mini.

        Args:
            h_steps: Passos horizontais (-2600 a 2600).
            v_steps: Passos verticais (-2600 a 2600).
        """
        h_steps = max(-2600, min(2600, h_steps))
        v_steps = max(-2600, min(2600, v_steps))

        # Execute motor_mini on camera via telnet/nc
        cmd = (
            f"echo '/tmp/m {h_steps} {v_steps}' "
            f"| nc -w 2 {self.motor_host} {self.motor_telnet_port}"
        )

        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=10.0
            )
            output = stdout.decode(errors="replace").strip()
            return f"Motor movido H={h_steps} V={v_steps}. Output: {output or 'ok'}"
        except TimeoutError:
            return "Timeout ao comunicar com camera."
        except Exception as e:
            logger.exception("Motor move failed")
            return f"Erro motor: {e}"
