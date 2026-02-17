from __future__ import annotations

import asyncio
import logging

from enton.core.tools import tool

logger = logging.getLogger(__name__)

# Camera ONVIF endpoint for digital PTZ
_ONVIF_HOST = "192.168.18.222"
_ONVIF_PORT = 5000

# Motor ioctl via telnet (physical PTZ)
_MOTOR_HOST = "192.168.18.222"
_MOTOR_TELNET_PORT = 23


async def _onvif_continuous_move(pan: float, tilt: float, duration: float = 1.0) -> str:
    """Send ONVIF ContinuousMove for digital PTZ."""

    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
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

    url = f"http://{_ONVIF_HOST}:{_ONVIF_PORT}/onvif/ptz_service"
    headers = {"Content-Type": "application/soap+xml; charset=utf-8"}

    try:
        proc = await asyncio.create_subprocess_exec(
            "curl", "-s", "-X", "POST", url,
            "-H", f"Content-Type: {headers['Content-Type']}",
            "-d", soap_body,
            "--connect-timeout", "3",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=5.0)

        await asyncio.sleep(duration)

        proc = await asyncio.create_subprocess_exec(
            "curl", "-s", "-X", "POST", url,
            "-H", f"Content-Type: {headers['Content-Type']}",
            "-d", soap_stop,
            "--connect-timeout", "3",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=5.0)

        return "OK"
    except Exception as e:
        return f"Erro ONVIF: {e}"


@tool
async def camera_move(direction: str, duration: float = 1.0) -> str:
    """Move a câmera na direção especificada usando PTZ digital (ONVIF).

    Args:
        direction: Direção do movimento: up, down, left, right, ou stop.
        duration: Duração do movimento em segundos (default 1.0).
    """
    direction = direction.lower().strip()

    pan_tilt = {
        "left": (-0.5, 0.0),
        "right": (0.5, 0.0),
        "up": (0.0, 0.5),
        "down": (0.0, -0.5),
        "esquerda": (-0.5, 0.0),
        "direita": (0.5, 0.0),
        "cima": (0.0, 0.5),
        "baixo": (0.0, -0.5),
    }

    if direction == "stop" or direction == "parar":
        result = await _onvif_continuous_move(0, 0, 0)
        return f"Câmera parada. {result}"

    if direction not in pan_tilt:
        return (
            f"Direção '{direction}' inválida. "
            "Use: up/down/left/right (ou cima/baixo/esquerda/direita)."
        )

    pan, tilt = pan_tilt[direction]
    result = await _onvif_continuous_move(pan, tilt, duration)
    return f"Câmera movida para {direction} por {duration}s. {result}"


@tool
async def camera_motor_move(h_steps: int = 0, v_steps: int = 0) -> str:
    """Move o motor físico da câmera (PTZ mecânico) via ioctl remoto.

    O driver multiplica steps por 10x internamente.
    Steps positivos = horário/cima, negativos = anti-horário/baixo.

    Args:
        h_steps: Passos horizontais (-2600 a 2600).
        v_steps: Passos verticais (-2600 a 2600).
    """
    h_steps = max(-2600, min(2600, h_steps))
    v_steps = max(-2600, min(2600, v_steps))

    # Execute motor_mini on camera via telnet
    # motor_mini binary must be present on camera at /tmp/m or /mnt/sdcard/motor_mini
    cmd = f"echo '/tmp/m {h_steps} {v_steps}' | nc -w 2 {_MOTOR_HOST} {_MOTOR_TELNET_PORT}"

    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        output = stdout.decode(errors="replace").strip()
        return f"Motor movido H={h_steps} V={v_steps}. Output: {output or 'ok'}"
    except TimeoutError:
        return "Timeout ao comunicar com câmera."
    except Exception as e:
        logger.exception("Motor move failed")
        return f"Erro motor: {e}"
