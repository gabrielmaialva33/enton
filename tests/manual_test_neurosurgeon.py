import os

from enton.skills.neurosurgeon_toolkit import NeurosurgeonToolkit


def test_neurosurgeon():
    print("=== TESTANDO NEUROSURGEON TOOLKIT ===")

    # Criar um arquivo dummy para cirurgia
    dummy_path = "/home/gabriel-maia/Documentos/enton/src/enton/dummy_yolo.py"
    with open(dummy_path, "w") as f:
        f.write("def useless_function():\n    print('Old code')\n    return False\n")

    print(f"Arquivo dummy criado em: {dummy_path}")

    toolkit = NeurosurgeonToolkit()

    # Teste 1: Ler Source
    print("\n--- Read Source (enton.dummy_yolo) ---")
    source = toolkit.read_enton_source("enton.dummy_yolo")
    print(source)
    if "Old code" not in source:
        print("FALHA: Não conseguiu ler o código corretamente.")

    # Teste 2: Backup
    print("\n--- Backup Module ---")
    backup_res = toolkit.backup_module("enton.dummy_yolo")
    print(backup_res)

    # Teste 3: Rewrite
    print("\n--- Rewrite Function ---")
    new_code_body = "def useless_function():\n    print('I AM EVOLVED')\n    return True"
    rewrite_res = toolkit.rewrite_function("enton.dummy_yolo", "useless_function", new_code_body)
    print(rewrite_res)

    # Verificar resultado
    with open(dummy_path) as f:
        new_content = f.read()
    print("\n--- Novo Conteúdo ---")
    print(new_content)

    if "I AM EVOLVED" in new_content:
        print("SUCESSO: O Código foi reescrito!")
    else:
        print("FALHA: O código não mudou.")

    # Limpeza
    if os.path.exists(dummy_path):
        os.remove(dummy_path)
    print("\n=== TESTE CONCLUÍDO ===")


if __name__ == "__main__":
    test_neurosurgeon()
