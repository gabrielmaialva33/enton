from enton.skills.god_mode_toolkit import GodModeToolkit
import time

def test_god_mode():
    print("=== TESTANDO GOD MODE TOOLKIT ===")
    
    toolkit = GodModeToolkit()
    tools = toolkit.get_tools()
    print(f"Ferramentas carregadas: {[t['name'] for t in tools]}")
    
    print("\n--- System Stats ---")
    stats = toolkit.system_stats()
    print(stats)
    
    print("\n--- List Heavy Processes (Memory) ---")
    heavy = toolkit.list_heavy_processes(limit=3, sort_by="memory")
    print(heavy)
    
    print("\n--- Judge Process (Self) ---")
    import os
    my_pid = os.getpid()
    judgment = toolkit.judge_process(my_pid)
    print(judgment)

    print("\n=== TESTE CONCLUÍDO ===")

if __name__ == "__main__":
    test_god_mode()
