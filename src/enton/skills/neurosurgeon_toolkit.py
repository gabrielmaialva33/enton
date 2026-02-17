import os
import ast
import inspect
import importlib
import logging
import shutil
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class NeurosurgeonToolkit:
    """
    Auto-Cirurgia Viva Toolkit.
    Permite ao Enton ler, modificar e recarregar seu próprio código.
    PODE QUEBRAR O AGENTE. USE COM EXTREMA CAUTELA.
    """
    def __init__(self, base_path: str = "/home/gabriel-maia/Documentos/enton/src"):
        self.name = "neurosurgeon_toolkit"
        self.base_path = base_path

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "read_enton_source",
                "description": "Lê o código fonte de um módulo do Enton.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "module_name": {"type": "string", "description": "Nome do módulo (ex: enton.app, enton.core.config)"}
                    },
                    "required": ["module_name"]
                }
            },
            {
                "name": "backup_module",
                "description": "Cria um backup de emergência de um arquivo de código antes de mexer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "module_name": {"type": "string", "description": "Nome do módulo a ser salvo."}
                    },
                    "required": ["module_name"]
                }
            },
            {
                "name": "rewrite_function",
                "description": "Substitui completamente uma função/método em um módulo.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "module_name": {"type": "string", "description": "Nome do módulo (ex: enton.core.memory)"},
                        "function_name": {"type": "string", "description": "Nome da função/classe a reescrever"},
                        "new_code": {"type": "string", "description": "O novo código completo da função/classe"}
                    },
                    "required": ["module_name", "function_name", "new_code"]
                }
            },
            {
                "name": "hot_reload",
                "description": "Tenta recarregar um módulo em memória (importlib.reload). Perigoso.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "module_name": {"type": "string", "description": "Nome do módulo a recarregar"}
                    },
                    "required": ["module_name"]
                }
            },
            {
                "name": "run_test_suite",
                "description": "Roda testes unitários para validar se a cirurgia deu certo.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "test_path": {"type": "string", "description": "Caminho do teste (ex: tests/core/test_memory.py)"}
                    },
                    "required": ["test_path"]
                }
            }
        ]

    def _get_file_path(self, module_name: str) -> str:
        # enton.core.config -> src/enton/core/config.py
        parts = module_name.split('.')
        # Assumindo que o primeiro part 'enton' já está em src/enton
        if parts[0] == 'enton':
            rel_path = os.path.join(*parts) + ".py"
            # base_path é .../src
            # rel_path é enton/dummy.py
            # full_path = .../src/enton/dummy.py
            full_path = os.path.join(self.base_path, rel_path) 
            return os.path.abspath(full_path)
        return ""

    def read_enton_source(self, module_name: str) -> str:
        file_path = self._get_file_path(module_name)
        if not os.path.exists(file_path):
            return f"Erro: Arquivo não encontrado para módulo {module_name} ({file_path})"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return f"--- CÓDIGO FONTE: {module_name} ---\n{content}"
        except Exception as e:
            return f"Erro ao ler arquivo: {e}"

    def backup_module(self, module_name: str) -> str:
        file_path = self._get_file_path(module_name)
        if not os.path.exists(file_path):
            return f"Erro: Arquivo original não existe."
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{file_path}.{timestamp}.bak"
        try:
            shutil.copy2(file_path, backup_path)
            return f"Backup criado com sucesso: {backup_path}"
        except Exception as e:
            return f"Falha ao criar backup: {e}"

    def rewrite_function(self, module_name: str, function_name: str, new_code: str) -> str:
        # AVISO: Isso é bruta força. Ideal seria CST. 
        # Aqui vamos tentar achar a def function e substituir bloco.
        
        file_path = self._get_file_path(module_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source)
            target_node = None
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.name == function_name:
                        target_node = node
                        break
            
            if not target_node:
                return f"Erro: Função/Classe '{function_name}' não encontrada no AST de {module_name}."

            # Localizar linhas (1-based)
            start_line = target_node.lineno - 1
            end_line = target_node.end_lineno
            
            lines = source.splitlines()
            
            # Manter indentação original da primeira linha
            original_indent = lines[start_line][:len(lines[start_line]) - len(lines[start_line].lstrip())]
            
            # Aplicar indentação no new_code
            new_lines_raw = new_code.strip().splitlines()
            new_lines_indented = [original_indent + line if line.strip() else line for line in new_lines_raw]
            
            # Substituir
            lines[start_line:end_line] = new_lines_indented
            
            new_source = "\n".join(lines)
            
            # Validar sintaxe antes de salvar
            try:
                ast.parse(new_source)
            except SyntaxError as se:
                return f"Erro de Sintaxe no novo código gerado! Abortando cirurgia: {se}"
            
            # Salvar
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_source)
                
            return f"Cirurgia realizada com sucesso em {module_name}.{function_name}. Arquivo salvo."

        except Exception as e:
            return f"Erro crítico na cirurgia: {e}"

    def hot_reload(self, module_name: str) -> str:
        try:
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
                return f"Módulo {module_name} recarregado em memória (Hot Reload)."
            else:
                return f"Módulo {module_name} não está carregado no sys.modules. Tente reiniciar o app."
        except Exception as e:
            return f"Falha no Hot Reload: {e}"

    def run_test_suite(self, test_path: str) -> str:
        import subprocess
        try:
            # pytest test_path
            cmd = ["pytest", test_path]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                return f"TESTES PASSARAM!\n{res.stdout[-500:]}" # Ultimas linhas
            else:
                return f"TESTES FALHARAM!\n{res.stdout[-1000:]}\nSTDERR:\n{res.stderr}"
        except Exception as e:
            return f"Erro ao rodar testes: {e}"
