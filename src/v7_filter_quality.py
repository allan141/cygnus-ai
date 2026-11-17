"""Cygnus v7 — auditoria heurística de qualidade textual."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ORIGEM = Path("data/v7_extraidos/wikiversity_20261001")
AUDITORIA = Path("reports/v7_auditoria_wikiversity.json")
SAIDA = Path("reports/v7_filtro_qualidade.json")

PALAVRAS_ESTRUTURAIS = {
    "de", "da", "do", "das", "dos", "em", "para",
    "com", "que", "uma", "um", "como", "por",
    "sobre", "entre", "também", "são", "ser",
}

TERMOS_ADMINISTRATIVOS = [
    "ementa:", "carga horária", "pré-requisitos",
    "objetivos específicos", "plano de ensino",
    "cronograma de aulas", "bibliografia básica",
]

TERMOS_SENSIVEIS = [
    "nome completo:", "cpf:", "rg:", "matrícula:",
    "entrevista com", "nome do aluno:", "turma:",
]


def avaliar(texto, tokens):
    motivos = []
    alertas = []

    linhas = [
        linha.strip()
        for linha in texto.splitlines()
        if linha.strip()
    ]

    palavras = re.findall(r"\b[\wÀ-ÿ]+\b", texto.lower())
    quantidade = len(palavras)

    if quantidade == 0:
        return "rejeitado", ["sem_palavras"], {}

    links = len(re.findall(r"https?://|www\.", texto, re.I))
    proporcao_links = links / max(quantidade / 100, 1)

    linhas_curtas = sum(
        1 for linha in linhas
        if len(linha) < 45
    )

    proporcao_linhas_curtas = (
        linhas_curtas / len(linhas) if linhas else 0
    )

    letras = sum(c.isalpha() for c in texto)
    proporcao_letras = letras / max(len(texto), 1)

    vocabulario = set(palavras)
    diversidade = len(vocabulario) / quantidade

    conectores = sum(
        1 for palavra in palavras
        if palavra in PALAVRAS_ESTRUTURAIS
    )
    proporcao_conectores = conectores / quantidade

    texto_minusculo = texto.lower()

    if tokens < 200:
        motivos.append("menos_200_tokens")

    if proporcao_letras < 0.45:
        motivos.append("poucas_letras")

    if links >= 15 and proporcao_links > 1.5:
        motivos.append("excesso_links")

    if quantidade >= 100 and proporcao_conectores < 0.045:
        motivos.append("pouca_linguagem_natural")

    if (
        len(linhas) >= 12
        and proporcao_linhas_curtas > 0.80
    ):
        motivos.append("predominio_linhas_curtas")

    if any(
        termo in texto_minusculo
        for termo in TERMOS_ADMINISTRATIVOS
    ):
        alertas.append("possivel_ementa")

    if any(
        termo in texto_minusculo
        for termo in TERMOS_SENSIVEIS
    ):
        alertas.append("possiveis_dados_pessoais")

    if links >= 5:
        alertas.append("contem_varios_links")

    if quantidade >= 100 and diversidade < 0.12:
        alertas.append("possivel_repeticao")

    metricas = {
        "palavras": quantidade,
        "links": links,
        "proporcao_letras": round(proporcao_letras, 4),
        "proporcao_linhas_curtas": round(
            proporcao_linhas_curtas, 4
        ),
        "diversidade_lexical": round(diversidade, 4),
        "proporcao_conectores": round(
            proporcao_conectores, 4
        ),
    }

    if motivos:
        return "rejeitado", motivos + alertas, metricas

    if alertas:
        return "revisao", alertas, metricas

    return "candidato", [], metricas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amostras", type=int, default=5)
    args = parser.parse_args()

    if not AUDITORIA.exists():
        raise FileNotFoundError(AUDITORIA)

    dados = json.loads(AUDITORIA.read_text(encoding="utf-8"))
    documentos = dados["documentos"]

    resultados = []
    categorias = Counter()
    tokens_por_categoria = Counter()
    motivos_contagem = Counter()

    for indice, documento in enumerate(documentos, 1):
        caminho = ORIGEM / "documentos" / documento["arquivo"]
        texto = caminho.read_text(encoding="utf-8")

        categoria, motivos, metricas = avaliar(
            texto, documento["tokens"]
        )

        registro = {
            "arquivo": documento["arquivo"],
            "tokens": documento["tokens"],
            "categoria": categoria,
            "motivos": motivos,
            "metricas": metricas,
        }

        resultados.append(registro)
        categorias[categoria] += 1
        tokens_por_categoria[categoria] += documento["tokens"]
        motivos_contagem.update(motivos)

        if indice % 1000 == 0:
            print(
                f"Analisados: {indice}/{len(documentos)}",
                flush=True,
            )

    relatorio = {
        "versao": "cygnus-v7-filtro-20.8",
        "metodo": "heuristicas_textuais",
        "categorias": dict(categorias),
        "tokens_por_categoria": dict(tokens_por_categoria),
        "motivos_frequentes": dict(
            motivos_contagem.most_common(20)
        ),
        "documentos": resultados,
        "observacao": (
            "Candidato não significa aprovado para treinamento. "
            "Exige revisão de qualidade, privacidade, licença "
            "e duplicatas aproximadas."
        ),
    }

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== CYGNUS V7 | FILTRO DE QUALIDADE ===")
    print("Documentos analisados:", len(resultados))

    for categoria in ("candidato", "revisao", "rejeitado"):
        print(
            f"{categoria}: "
            f"{categorias[categoria]} documentos | "
            f"{tokens_por_categoria[categoria]:,} tokens"
        )

    print("\nMotivos mais frequentes:")
    for motivo, total in motivos_contagem.most_common(10):
        print(f"  {motivo}: {total}")

    for categoria in ("candidato", "revisao", "rejeitado"):
        exemplos = [
            item for item in resultados
            if item["categoria"] == categoria
        ][:args.amostras]

        print(f"\n=== EXEMPLOS: {categoria.upper()} ===")

        for item in exemplos:
            caminho = ORIGEM / "documentos" / item["arquivo"]
            texto = caminho.read_text(encoding="utf-8")
            trecho = re.sub(r"\s+", " ", texto[:250])

            print("Arquivo:", item["arquivo"])
            print("Tokens:", item["tokens"])
            print("Motivos:", item["motivos"])
            print("Trecho:", trecho)
            print()

    print("Relatório:", SAIDA)
    print("FILTRAGEM CONCLUÍDA")


if __name__ == "__main__":
    main()
