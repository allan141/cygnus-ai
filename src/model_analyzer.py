
# Cygnus AI - Fase 15
# Analise de arquitetura e memoria

import torch

from src.language_model import CygnusLanguageModel
from src.model_config import ConfiguracaoCygnus


def analisar_modelo(
    tamanho_vocabulario=512,
    configuracao=None
):
    if configuracao is None:
        configuracao = ConfiguracaoCygnus()

    modelo = CygnusLanguageModel(
        **configuracao.para_modelo(
            tamanho_vocabulario
        )
    )

    parametros = sum(
        parametro.numel()
        for parametro in modelo.parameters()
    )

    treinaveis = sum(
        parametro.numel()
        for parametro in modelo.parameters()
        if parametro.requires_grad
    )

    bytes_parametros = sum(
        parametro.numel() * parametro.element_size()
        for parametro in modelo.parameters()
    )

    return {
        "modelo": modelo,
        "parametros": parametros,
        "treinaveis": treinaveis,
        "memoria_pesos_mb": bytes_parametros / (1024 ** 2)
    }


def testar_forward(modelo, tamanho_contexto=16):
    modelo.eval()

    vocabulario = modelo.tamanho_vocabulario

    entradas = torch.randint(
        0,
        vocabulario,
        (1, tamanho_contexto),
        dtype=torch.long
    )

    with torch.no_grad():
        logits, _ = modelo(entradas)

    return logits.shape


if __name__ == "__main__":
    resultado = analisar_modelo()

    modelo = resultado["modelo"]

    print("\n=== CYGNUS AI v3 ===")
    print("Parametros:", resultado["parametros"])
    print("Treinaveis:", resultado["treinaveis"])

    print(
        "Memoria dos pesos:",
        f"{resultado['memoria_pesos_mb']:.2f} MiB"
    )

    print(
        "Formato da saida:",
        testar_forward(modelo)
    )
