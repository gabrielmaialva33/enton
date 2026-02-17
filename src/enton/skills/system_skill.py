import datetime
import logging
import platform

import psutil

from enton.core.tools import tool

logger = logging.getLogger(__name__)

@tool
def get_system_stats() -> str:
    """Retorna estatísticas atuais do sistema (CPU, Memória, Disco, Bateria)."""
    cpu = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    stats = [
        f"OS: {platform.system()} {platform.release()}",
        f"CPU Usage: {cpu}%",
        f"Memory: {mem.percent}% used ({mem.used // (1024**3)}GB / {mem.total // (1024**3)}GB)",
        f"Disk: {disk.percent}% used",
    ]
    
    if hasattr(psutil, "sensors_battery"):
        battery = psutil.sensors_battery()
        if battery:
            charging = "(Charging)" if battery.power_plugged else ""
            stats.append(f"Battery: {battery.percent}% {charging}")
            
    return "\n".join(stats)

@tool
def get_time() -> str:
    """Retorna a data e hora atual no formato ISO."""
    return datetime.datetime.now().isoformat()

@tool
def list_processes(limit: int = 5) -> str:
    """Lista os processos que mais consomem CPU.
    
    Args:
        limit: Número de processos a listar (default: 5).
    """
    procs = []
    for p in psutil.process_iter(['pid', 'name', 'cpu_percent']):
        try:
            procs.append(p.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
            
    # Sort by cpu_percent
    procs.sort(key=lambda x: x['cpu_percent'], reverse=True)
    
    summary = []
    for p in procs[:limit]:
        summary.append(f"PID {p['pid']}: {p['name']} ({p['cpu_percent']}%)")
        
    return "\n".join(summary)
