import os
from pathlib import Path

def gerar_arvore(diretorio, ignorar="", prefixo=""):
    path = Path(diretorio)
    
    # Lista e filtra o conteúdo (ignora a pasta escolhida e arquivos ocultos comuns)
    try:
        # Pega todos os itens e remove o que deve ser ignorado
        itens = sorted([item for item in path.iterdir() if item.name != ignorar])
    except PermissionError:
        return # Pula pastas que o Python não tem permissão de ler

    for i, item in enumerate(itens):
        extensao = "└── " if i == len(itens) - 1 else "├── "
        print(f"{prefixo}{extensao}{item.name}")

        if item.is_dir():
            # Define o novo recuo visual
            proximo_prefixo = prefixo + ("    " if i == len(itens) - 1 else "│   ")
            # Chamada recursiva
            gerar_arvore(item, ignorar, proximo_prefixo)

if __name__ == "__main__":
    raiz = os.getcwd()
    print(f"Estrutura de: {raiz}")
    
    # Exemplo: se digitar 'venv' ou '.git', ele não entrará nessas pastas
    pasta_para_pular = input("node_modules").strip()
    
    print(".")
    gerar_arvore(raiz, ignorar=pasta_para_pular)