"""NetworkTools — Agno toolkit for network scanning and device control.

Enton can scan the network (nmap), discover devices, control Bluetooth
(bluetoothctl), manage Tailscale VPN, and check connectivity.

Requires: nmap, bluetoothctl, tailscale (all pre-installed).
"""

from __future__ import annotations

import logging
import subprocess

from agno.tools import Toolkit

logger = logging.getLogger(__name__)


class NetworkTools(Toolkit):
    """Tools for network scanning, Bluetooth, and connectivity."""

    def __init__(self) -> None:
        super().__init__(name="network_tools")
        self.register(self.network_scan)
        self.register(self.port_scan)
        self.register(self.ping)
        self.register(self.local_ip)
        self.register(self.bluetooth_devices)
        self.register(self.bluetooth_connect)
        self.register(self.bluetooth_disconnect)
        self.register(self.tailscale_status)
        self.register(self.wifi_networks)
        self.register(self.dns_lookup)

    def network_scan(self, subnet: str = "192.168.18.0/24") -> str:
        """Quick scan of local network to find active devices.

        Args:
            subnet: Network to scan (default: 192.168.18.0/24)
        """
        try:
            result = subprocess.run(
                ["nmap", "-sn", "-T4", subnet],
                timeout=30,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip() or "Nenhum host encontrado."
        except subprocess.TimeoutExpired:
            return "Timeout no scan (30s)"
        except Exception as e:
            return f"Erro nmap: {e}"

    def port_scan(self, host: str, ports: str = "1-1000") -> str:
        """Scan ports on a specific host.

        Args:
            host: IP or hostname to scan
            ports: Port range (e.g., "22,80,443" or "1-1000")
        """
        try:
            result = subprocess.run(
                ["nmap", "-T4", "-p", ports, host],
                timeout=60,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            return "Timeout (60s)"
        except Exception as e:
            return f"Erro: {e}"

    def ping(self, host: str, count: int = 4) -> str:
        """Ping a host to check connectivity.

        Args:
            host: IP or hostname
            count: Number of pings (default: 4)
        """
        try:
            result = subprocess.run(
                ["ping", "-c", str(min(count, 10)), "-W", "2", host],
                timeout=15,
                capture_output=True,
                text=True,
            )
            # Extract summary
            lines = result.stdout.strip().split("\n")
            return "\n".join(lines[-3:]) if len(lines) > 3 else result.stdout.strip()
        except subprocess.TimeoutExpired:
            return f"Timeout: {host} nao respondeu"
        except Exception as e:
            return f"Erro: {e}"

    def local_ip(self) -> str:
        """Show all local IP addresses and network interfaces."""
        try:
            result = subprocess.run(
                ["ip", "-br", "addr"],
                timeout=3,
                capture_output=True,
                text=True,
            )
            lines = [
                line for line in result.stdout.strip().split("\n") if not line.startswith("lo ")
            ]
            return "\n".join(lines) if lines else "Nenhuma interface de rede."
        except Exception as e:
            return f"Erro: {e}"

    def bluetooth_devices(self) -> str:
        """List paired and nearby Bluetooth devices."""
        try:
            # Paired devices
            paired = subprocess.run(
                ["bluetoothctl", "devices"],
                timeout=5,
                capture_output=True,
                text=True,
            ).stdout.strip()

            # Connected devices
            connected = subprocess.run(
                ["bluetoothctl", "devices", "Connected"],
                timeout=5,
                capture_output=True,
                text=True,
            ).stdout.strip()

            result = "Pareados:\n" + (paired or "  (nenhum)")
            if connected:
                result += "\n\nConectados:\n" + connected
            return result
        except Exception as e:
            return f"Erro: {e}"

    def bluetooth_connect(self, mac: str) -> str:
        """Connect to a Bluetooth device by MAC address.

        Args:
            mac: Device MAC address (e.g., "90:B6:85:19:74:B9")
        """
        try:
            result = subprocess.run(
                ["bluetoothctl", "connect", mac],
                timeout=15,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip() or result.stderr.strip()
        except subprocess.TimeoutExpired:
            return f"Timeout conectando {mac}"
        except Exception as e:
            return f"Erro: {e}"

    def bluetooth_disconnect(self, mac: str) -> str:
        """Disconnect a Bluetooth device.

        Args:
            mac: Device MAC address
        """
        try:
            result = subprocess.run(
                ["bluetoothctl", "disconnect", mac],
                timeout=10,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip() or result.stderr.strip()
        except Exception as e:
            return f"Erro: {e}"

    def tailscale_status(self) -> str:
        """Show Tailscale VPN status and connected peers."""
        try:
            result = subprocess.run(
                ["tailscale", "status"],
                timeout=10,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip() or "Tailscale nao conectado."
        except Exception as e:
            return f"Erro: {e}"

    def wifi_networks(self) -> str:
        """Scan for nearby WiFi networks."""
        try:
            result = subprocess.run(
                ["nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY", "device", "wifi", "list"],
                timeout=15,
                capture_output=True,
                text=True,
            )
            lines = result.stdout.strip().split("\n")
            formatted = ["  SSID | Signal | Security"]
            formatted.append("  " + "-" * 40)
            for line in lines[:20]:
                parts = line.split(":")
                if len(parts) >= 3:
                    ssid, signal, security = parts[0], parts[1], parts[2]
                    formatted.append(f"  {ssid:20s} | {signal:6s}% | {security}")
            return "\n".join(formatted)
        except Exception as e:
            return f"Erro: {e}"

    def dns_lookup(self, domain: str) -> str:
        """DNS lookup for a domain.

        Args:
            domain: Domain name to resolve
        """
        try:
            result = subprocess.run(
                ["dig", "+short", domain],
                timeout=10,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip() or f"Nenhum registro para {domain}"
        except Exception as e:
            # Fallback
            try:
                result = subprocess.run(
                    ["nslookup", domain],
                    timeout=10,
                    capture_output=True,
                    text=True,
                )
                return result.stdout.strip()
            except Exception:
                return f"Erro DNS: {e}"
