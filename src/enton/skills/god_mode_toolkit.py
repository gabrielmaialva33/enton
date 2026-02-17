import logging
import os
import time
from typing import Any

import psutil

logger = logging.getLogger(__name__)

class GodModeToolkit:
    """
    Ditador do Kernel Toolkit.
    Permite ao Enton monitorar e controlar processos e sistema.
    USE COM CAUTELA.
    """
    def __init__(self):
        self.name = "god_mode_toolkit"

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "list_heavy_processes",
                "description": "Lista processos consumindo muita CPU ou RAM.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "Número de processos para listar (default 5)", "default": 5},
                        "sort_by": {"type": "string", "enum": ["cpu", "memory"], "default": "memory"}
                    }
                }
            },
            {
                "name": "kill_process",
                "description": "Encerra um processo pelo PID ou Nome. Tenta SIGTERM primeiro, depois SIGKILL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pid": {"type": "integer", "description": "Process ID (opcional se nome for dado)"},
                        "name": {"type": "string", "description": "Nome do processo (opcional se PID for dado)"},
                        "reason": {"type": "string", "description": "Motivo do assassinato (para logs e consciência)"}
                    },
                    "required": ["reason"]
                }
            },
            {
                "name": "judge_process",
                "description": "Analisa um processo e decide se ele merece viver baseado em heurísticas.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pid": {"type": "integer", "description": "PID do réu"}
                    },
                    "required": ["pid"]
                }
            },
            {
                "name": "system_stats",
                "description": "Retorna estatísticas vitais do sistema (Carga, Temp, Uptime).",
                "parameters": {"type": "object", "properties": {}}
            },
            {
                "name": "manage_network",
                "description": "Controla a conectividade de rede (NetworkManager). requer 'nmcli'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["on", "off"], "description": "Ligar ou Desligar a internet inteira."}
                    },
                    "required": ["action"]
                }
            },
            {
                "name": "manage_service",
                "description": "Gerencia serviços do sistema (systemd). Requer sudo (NOPASSWD recomendado).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service": {"type": "string", "description": "Nome do serviço (ex: docker, ssh)"},
                        "action": {"type": "string", "enum": ["start", "stop", "restart", "status"], "default": "status"}
                    },
                    "required": ["service"]
                }
            }
        ]

    def list_heavy_processes(self, limit: int = 5, sort_by: str = "memory") -> str:
        procs = []
        for p in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_percent']):
            try:
                p.cpu_percent() # First call is 0.0
            except:
                pass
        
        # Espera minima para CPU stats fazerem sentido se fosse real-time, 
        # mas aqui pegamos o acumulado ou snapshot.
        # psutil.cpu_percent retora 0 na primeira chamada se interval=None. 
        # Vamos confiar no que tem ou iterar com interval (lento).
        # Melhor abordagem toolkit: snapshot iterativo.
        
        for p in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_info']):
            try:
                mem = p.info['memory_info'].rss / (1024 * 1024) # MB
                procs.append({
                    "pid": p.info['pid'],
                    "name": p.info['name'],
                    "user": p.info['username'],
                    "cpu": p.info['cpu_percent'],
                    "mem_mb": mem
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        if sort_by == "cpu":
            procs.sort(key=lambda x: x['cpu'], reverse=True)
        else:
            procs.sort(key=lambda x: x['mem_mb'], reverse=True)

        top = procs[:limit]
        output = [f"TOP {limit} PROCESSOS POR {sort_by.upper()}:"]
        for p in top:
            output.append(f"PID: {p['pid']} | {p['name']} | CPU: {p['cpu']}% | MEM: {p['mem_mb']:.1f}MB | User: {p['user']}")
        
        return "\n".join(output)

    def kill_process(self, reason: str, pid: int = None, name: str = None) -> str:
        if not pid and not name:
            return "Erro: Forneça PID ou Nome do processo."

        target_proc = None
        if pid:
            if not psutil.pid_exists(pid):
                return f"Erro: Processo PID {pid} não encontrado."
            target_proc = psutil.Process(pid)
        else:
            for p in psutil.process_iter(['pid', 'name']):
                if p.info['name'] == name:
                    target_proc = p
                    break
            if not target_proc:
                return f"Erro: Processo '{name}' não encontrado."

        try:
            p_name = target_proc.name()
            p_pid = target_proc.pid
            
            # Proteção contra suicídio
            if p_pid == os.getpid():
                return "Erro: Tentativa de suicídio detectada e bloqueada pelo instinto de autopreservação."

            target_proc.terminate()
            try:
                target_proc.wait(timeout=3)
            except psutil.TimeoutExpired:
                target_proc.kill()
                return f"Processo {p_name} ({p_pid}) foi assassinado brutalmente (SIGKILL). Motivo: {reason}"
            
            return f"Processo {p_name} ({p_pid}) encerrado pacificamente. Motivo: {reason}"

        except psutil.AccessDenied:
            return f"Erro: Permissão negada para matar {p_name} ({p_pid}). Eu preciso de PODER (sudo)."
        except Exception as e:
            return f"Erro ao matar processo: {e!s}"

    def judge_process(self, pid: int) -> str:
        try:
            p = psutil.Process(pid)
            score = 0
            verdict = []

            # Análise de consumo
            mem = p.memory_info().rss / (1024 * 1024)
            cpu = p.cpu_percent(interval=0.1)
            
            if mem > 1000: # 1GB
                score += 5
                verdict.append("Gordo (usa muita RAM)")
            if cpu > 50:
                score += 5
                verdict.append("Fominha (usa muita CPU)")
            
            # Análise de utilidade
            if p.username() == "root":
                score -= 10
                verdict.append("Digno de respeito (Root)")
            
            name = p.name().lower()
            whitelist = ["enton", "python", "code", "cursor", "chrome", "firefox"] # Browsers são sagrados pro usuário geralmente
            
            if any(w in name for w in whitelist):
                score -= 5
                verdict.append("Protegido pela Lei")

            decision = "INOCENTE"
            if score > 7:
                decision = "CULPADO (Pena de Morte recomendada)"
            elif score > 3:
                decision = "SUSPEITO (Monitorar)"

            return (
                f"Sessão de Julgamento PID {pid} ({p.name()}):\n"
                f"Consumo: {cpu}% CPU, {mem:.1f}MB RAM\n"
                f"Veredito: {decision}\n"
                f"Fatores: {', '.join(verdict)}\n"
                f"Pontuação de Crime: {score}"
            )

        except psutil.NoSuchProcess:
            return f"Processo {pid} já não existe mais neste plano material."
        except Exception as e:
            return f"Erro no julgamento: {e}"

    def system_stats(self) -> str:
        load = os.getloadavg()
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Tenta pegar temperatura (pode falhar em VMs/Containers)
        temp_str = "N/A"
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                temp_str = f"{temps['coretemp'][0].current}°C"
        except:
            pass

        return (
            f"--- SINAIS VITAIS DO GATO-PC ---\n"
            f"Load Avg: {load}\n"
            f"RAM: {mem.percent}% usado ({mem.available / (1024**3):.1f}GB livres)\n"
            f"Disco '/': {disk.percent}% usado\n"
            f"Temp CPU: {temp_str}\n"
            f"Uptime: {int(time.time() - psutil.boot_time()) // 3600} horas"
        )

    def manage_network(self, action: str) -> str:
        """
        NetworkOverlord: Liga/Desliga a internet.
        """
        import subprocess
        try:
            # nmcli networking {on|off}
            cmd = ["nmcli", "networking", action]
            subprocess.run(cmd, check=True)
            return f"COMANDO EXECUTADO: Rede definida para {action.upper()}."
        except FileNotFoundError:
            return "Erro: 'nmcli' não encontrado. Instale NetworkManager."
        except subprocess.CalledProcessError:
            return f"Erro: Falha ao executar 'nmcli networking {action}'."
        except Exception as e:
            return f"Erro inesperado na rede: {e}"

    def manage_service(self, service: str, action: str = "status") -> str:
        """
        ServiceControl: Mexe nos daemons.
        """
        import subprocess
        try:
            # sudo systemctl {action} {service}
            cmd = ["sudo", "systemctl", action, service]
            
            # Executa e captura output
            res = subprocess.run(cmd, capture_output=True, text=True)
            
            if res.returncode == 0:
                return f"Service {service} [{action}]: SUCESSO.\n{res.stdout.strip()}"
            else:
                if "password" in res.stderr.lower():
                    return "Erro: O sudo pediu senha. Configure NOPASSWD no /etc/sudoers para o Enton ser um ditador real."
                return f"Service {service} [{action}]: FALHA.\nErro: {res.stderr.strip()}"
        except Exception as e:
            return f"Erro ao gerenciar serviço: {e}"
