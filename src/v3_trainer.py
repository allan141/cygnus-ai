
# Cygnus AI - Fase 15, Parte 6
# Treinamento experimental do Transformer v3

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from src.advanced_trainer import (
    avaliar,
    perplexidade,
    registrar_metricas,
    salvar_checkpoint,
    carregar_checkpoint,
)
from src.byte_bpe_tokenizer import CygnusByteBPETokenizer
from src.language_model import CygnusLanguageModel
from src.model_config import ConfiguracaoCygnus
from src.corpus_analyzer import carregar_documentos



def preparar_sequencias(diretorio, tokenizador, contexto):
    documentos = carregar_documentos(diretorio)
    sequencias = []

    for documento in documentos:
        ids = tokenizador.codificar(documento["texto"])

        if len(ids) > contexto:
            sequencia = torch.tensor(
                ids,
                dtype=torch.long
            )

            sequencias.append(sequencia)

    if not sequencias:
        raise ValueError(
            "Nenhum documento possui tokens suficientes "
            "para o contexto solicitado."
        )

    return sequencias



def assinatura_corpus(diretorio):
    documentos = carregar_documentos(diretorio)
    hash_total = hashlib.sha256()

    for documento in documentos:
        caminho = Path(documento["arquivo"])
        hash_total.update(
            caminho.name.encode("utf-8")
        )
        hash_total.update(b"\x00")
        hash_total.update(
            documento["texto"].encode("utf-8")
        )
        hash_total.update(b"\x00")

    return hash_total.hexdigest()


def treinar_v3(args):
    if args.passos <= 0:
        raise ValueError("Passos deve ser positivo")
    if args.batch_size <= 0:
        raise ValueError("Batch size deve ser positivo")
    if args.avaliar_cada <= 0:
        raise ValueError("Intervalo de avaliacao invalido")
    if args.lr <= 0:
        raise ValueError("Taxa de aprendizado invalida")

    random.seed(args.semente)
    torch.manual_seed(args.semente)

    configuracao_base = ConfiguracaoCygnus()
    contexto = configuracao_base.tamanho_contexto

    pasta_corpus = Path(args.corpus)
    pasta_tokenizador = Path(args.tokenizador)
    pasta_experimento = Path(args.experimento)

    caminho_modelo = pasta_experimento / "modelo.pt"
    caminho_metricas = pasta_experimento / "metricas.csv"
    caminho_manifesto = pasta_experimento / "manifesto.json"

    tokenizador = CygnusByteBPETokenizer.carregar(
        pasta_tokenizador
    )

    configuracao = configuracao_base.para_modelo(
        tokenizador.tamanho_vocabulario
    )

    treino = preparar_sequencias(
        pasta_corpus / "train",
        tokenizador,
        contexto
    )

    validacao = preparar_sequencias(
        pasta_corpus / "validation",
        tokenizador,
        contexto
    )

    manifesto = {
        "versao": "cygnus-v3",
        "configuracao": configuracao,
        "tokenizador_sha256": hashlib.sha256(
            pasta_tokenizador.read_bytes()
        ).hexdigest(),
        "treino_sha256": assinatura_corpus(
            pasta_corpus / "train"
        ),
        "validacao_sha256": assinatura_corpus(
            pasta_corpus / "validation"
        ),
        "semente": args.semente,
        "lr": args.lr,
        "batch_size": args.batch_size
    }

    if args.retomar:
        if not caminho_modelo.exists():
            raise FileNotFoundError(
                f"Checkpoint inexistente: {caminho_modelo}"
            )

        if not caminho_manifesto.exists():
            raise FileNotFoundError(
                "Manifesto do experimento nao encontrado"
            )

        manifesto_anterior = json.loads(
            caminho_manifesto.read_text(encoding="utf-8")
        )

        if manifesto_anterior != manifesto:
            raise ValueError(
                "Configuracao, corpus ou tokenizador mudou. "
                "Nao e seguro retomar este experimento."
            )
    else:
        if any(
            caminho.exists()
            for caminho in (
                caminho_modelo,
                caminho_metricas,
                caminho_manifesto
            )
        ):
            raise FileExistsError(
                "O experimento ja existe. Escolha outro "
                "diretorio ou use --retomar."
            )

    dispositivo = torch.device("cpu")

    modelo = CygnusLanguageModel(
        **configuracao
    ).to(dispositivo)

    otimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=args.lr
    )

    passo_inicial = 0

    if args.retomar:
        checkpoint = torch.load(
            caminho_modelo,
            map_location="cpu",
            weights_only=True
        )

        if checkpoint["configuracao"] != configuracao:
            raise ValueError(
                "Arquitetura do checkpoint incompatível"
            )

        passo_inicial = carregar_checkpoint(
            caminho_modelo,
            modelo,
            otimizador
        )
    else:
        pasta_experimento.mkdir(
            parents=True,
            exist_ok=True
        )

        caminho_manifesto.write_text(
            json.dumps(
                manifesto,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

    parametros = sum(
        p.numel()
        for p in modelo.parameters()
        if p.requires_grad
    )

    print("\n=== CYGNUS AI - TREINAMENTO V3 ===")
    print("Dispositivo:", dispositivo)
    print("Parametros:", parametros)
    print("Vocabulario:", tokenizador.tamanho_vocabulario)
    print("Documentos treino:", len(treino))
    print("Documentos validacao:", len(validacao))
    print("Contexto:", contexto)
    print("Passo inicial:", passo_inicial)
    print("Passo final:", args.passos)

    if args.passos <= passo_inicial:
        print("Nenhum passo adicional necessario.")
        return

    # Utilizamos o mesmo amostrador do treinador v2.
    from src.dataset import criar_lote_documentos

    modelo.train()

    try:
        for passo in range(
            passo_inicial + 1,
            args.passos + 1
        ):
            x, y = criar_lote_documentos(
                treino,
                tamanho_contexto=contexto,
                tamanho_lote=args.batch_size
            )

            x = x.to(dispositivo)
            y = y.to(dispositivo)

            _, perda = modelo(x, y)

            otimizador.zero_grad()
            perda.backward()

            torch.nn.utils.clip_grad_norm_(
                modelo.parameters(),
                max_norm=1.0
            )

            otimizador.step()

            if (
                passo == 1
                or passo % args.avaliar_cada == 0
                or passo == args.passos
            ):
                perda_treino = avaliar(
                    modelo,
                    treino,
                    contexto,
                    args.batch_size,
                    lotes=3
                )

                perda_validacao = avaliar(
                    modelo,
                    validacao,
                    contexto,
                    args.batch_size,
                    lotes=3
                )

                print(
                    f"Passo {passo:05d} | "
                    f"Treino: {perda_treino:.4f} | "
                    f"Validacao: {perda_validacao:.4f} | "
                    f"PPL: {perplexidade(perda_validacao):.2f}",
                    flush=True
                )

                registrar_metricas(
                    caminho_metricas,
                    passo,
                    perda_treino,
                    perda_validacao
                )

                salvar_checkpoint(
                    caminho_modelo,
                    modelo,
                    otimizador,
                    passo,
                    configuracao
                )

    except KeyboardInterrupt:
        print(
            "\nTreinamento interrompido. "
            "Use --retomar para continuar."
        )
        return

    print("\nTreinamento v3 concluido.")
    print("Modelo:", caminho_modelo)
    print("Metricas:", caminho_metricas)


def main():
    parser = argparse.ArgumentParser(
        description="Treinador Cygnus AI v3"
    )

    parser.add_argument(
        "--corpus",
        default="data/v3_livros"
    )
    parser.add_argument(
        "--tokenizador",
        default="checkpoints/cygnus_v3/tokenizador.json"
    )
    parser.add_argument(
        "--experimento",
        default="checkpoints/cygnus_v3_treino"
    )
    parser.add_argument(
        "--passos",
        type=int,
        default=100
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4
    )
    parser.add_argument(
        "--avaliar-cada",
        type=int,
        default=25
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0003
    )
    parser.add_argument(
        "--semente",
        type=int,
        default=42
    )
    parser.add_argument(
        "--retomar",
        action="store_true"
    )

    args = parser.parse_args()
    treinar_v3(args)


if __name__ == "__main__":
    main()
