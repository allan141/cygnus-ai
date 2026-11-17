"""Cygnus v7 — filtro de qualidade v2."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ORIGEM = Path("data/v7_extraidos/wikiversity_20261001/documentos")
RELATORIO_V1 = Path("reports/v7_filtro_qualidade.json")
SAIDA = Path("reports/v7_filtro_qualidade_v2.json")

RESIDUOS_WIKI = [
    r"__NOTOC__",
    r"__TOC__",
    r"__NOEDITSECTION__",
    r"\b(?:thumb|thumbnail)\s*\|",
    r"\b(?:left|right|center)\s*\|\s*\d+px",
    r"\b(?:left|right|center)\s*\|\s*",
]

PADROES_NAVEGACAO = [
    r"\bbem[\s-]*vind[oa]s?\s+(?:ao?|à|aos?)\s+(?:curso|portal|instituto|preparat[oó]rio)",
    r"\bportal\s+d[aeo]s?\s+",
    r"\bgrade curricular\b",
    r"\blista de disciplinas\b",
    r"\bcursos e aulas\b",
    r"\bdepartamentos e cursos\b",
    r"\bnavegue pelas\b",
]

PADROES_EXPLICATIVOS = [
    r"\b(?:é definido como|é conhecida? como|significa|consiste em)\b",
    r"\b(?:por exemplo|como exemplo|um exemplo)\b",
    r"\b(?:isso ocorre porque|isso acontece porque|devido a)\b",
    r"\b(?:dessa forma|portanto|consequentemente|assim sendo)\b",
    r"\b(?:em outras palavras|ou seja|em resumo)\b",
]

PADROES_PRIVACIDADE = [
    r"\bcpf\s*[:\-]?\s*\d",
    r"\b(?:rg|matr[ií]cula)\s*[:\-]\s*\d",
    r"\bnome\s+(?:completo|do aluno)\s*:",
    r"\bentrevista\s+(?:com|sobre mem[oó]rias)\b",
    r"\bnome\s*:\s*[A-ZÀ-Ý][a-zà-ÿ]+\s+[A-ZÀ-Ý][a-zà-ÿ]+",
]

PADROES_ADMINISTRATIVOS = [
    r"\bementa\s*:",
    r"\bcarga hor[aá]ria\b",
    r"\bplano de ensino\b",
    r"\bbibliografia b[aá]sica\b",
    r"\bcronograma de aulas\b",
]


def limpar_residuos(texto):
    limpo = texto

    for padrao in RESIDUOS_WIKI:
        limpo = re.sub(padrao, " ", limpo, flags=re.I)

    limpo = re.sub(r"[ \t]+", " ", limpo)
    limpo = re.sub(r"\n{3,}", "\n\n", limpo)

    return limpo.strip()


def contar_padroes(texto, padroes):
    return sum(
        bool(re.search(padrao, texto, flags=re.I))
        for padrao in padroes
    )


def avaliar_v2(texto, registro_v1):
    texto_limpo = limpar_residuos(texto)

    palavras = re.findall(r"\b[\wÀ-ÿ]+\b", texto_limpo)
    paragrafos = [
        p.strip()
        for p in re.split(r"\n\s*\n", texto_limpo)
        if p.strip()
    ]

    paragrafos_longos = sum(
        len(p) >= 180 for p in paragrafos
    )

    navegacao = contar_padroes(
        texto_limpo[:1800], PADROES_NAVEGACAO
    )

    explicacao = contar_padroes(
        texto_limpo, PADROES_EXPLICATIVOS
    )

    privacidade = contar_padroes(
        texto_limpo, PADROES_PRIVACIDADE
    )

    administrativo = contar_padroes(
        texto_limpo, PADROES_ADMINISTRATIVOS
    )

    motivos = []
    categoria_v1 = registro_v1["categoria"]
    motivos_v1 = set(registro_v1["motivos"])

    # Sinais sensíveis prevalecem sobre pontuação textual.
    if privacidade:
        motivos.append("possiveis_dados_pessoais_v2")

    if "possiveis_dados_pessoais" in motivos_v1:
        motivos.append("alerta_privacidade_v1")

    if navegacao >= 2:
        motivos.append("possivel_pagina_navegacao")
    elif navegacao == 1 and explicacao == 0:
        motivos.append("possivel_pagina_navegacao")

    if administrativo >= 2:
        motivos.append("possivel_material_administrativo")

    if len(palavras) < 120:
        motivos.append("pouco_conteudo_textual")

    if (
        len(paragrafos) >= 5
        and paragrafos_longos == 0
    ):
        motivos.append("sem_paragrafos_desenvolvidos")

    # Rejeições estruturais anteriores continuam conservadas.
    motivos_rejeicao_v1 = {
        "menos_200_tokens",
        "poucas_letras",
        "excesso_links",
        "pouca_linguagem_natural",
        "predominio_linhas_curtas",
    }

    rejeicao_estrutural = bool(
        motivos_v1 & motivos_rejeicao_v1
    )

    if rejeicao_estrutural:
        motivos.append("problema_estrutural_v1")

    if privacidade or "possiveis_dados_pessoais" in motivos_v1:
        categoria = "revisao"
    elif rejeicao_estrutural:
        categoria = "rejeitado"
    elif motivos:
        categoria = "revisao"
    elif categoria_v1 == "revisao":
        # Não promover automaticamente documentos já sinalizados.
        categoria = "revisao"
        motivos.append("alerta_pendente_v1")
    else:
        categoria = "candidato"

    metricas = {
        "palavras": len(palavras),
        "paragrafos": len(paragrafos),
        "paragrafos_longos": paragrafos_longos,
        "sinais_navegacao": navegacao,
        "sinais_explicacao": explicacao,
        "sinais_privacidade": privacidade,
        "sinais_administrativos": administrativo,
        "caracteres_apos_limpeza": len(texto_limpo),
    }

    return categoria, motivos, metricas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amostras", type=int, default=5)
    args = parser.parse_args()

    dados = json.loads(
        RELATORIO_V1.read_text(encoding="utf-8")
    )

    resultados = []
    categorias = Counter()
    tokens_categorias = Counter()
    transicoes = Counter()
    motivos = Counter()

    for indice, registro in enumerate(dados["documentos"], 1):
        caminho = ORIGEM / registro["arquivo"]
        texto = caminho.read_text(encoding="utf-8")

        categoria, alertas, metricas = avaliar_v2(
            texto, registro
        )

        novo = {
            "arquivo": registro["arquivo"],
            "tokens": registro["tokens"],
            "categoria_v1": registro["categoria"],
            "categoria_v2": categoria,
            "motivos_v1": registro["motivos"],
            "motivos_v2": alertas,
            "metricas_v2": metricas,
            "status_treinamento": "nao_aprovado",
        }

        resultados.append(novo)

        categorias[categoria] += 1
        tokens_categorias[categoria] += registro["tokens"]

        transicoes[
            f"{registro['categoria']} -> {categoria}"
        ] += 1

        motivos.update(alertas)

        if indice % 1000 == 0:
            print(
                f"Analisados: {indice}/{len(dados['documentos'])}",
                flush=True,
            )

    relatorio = {
        "versao": "cygnus-v7-filtro-20.9",
        "documentos_analisados": len(resultados),
        "categorias": dict(categorias),
        "tokens_por_categoria": dict(tokens_categorias),
        "transicoes": dict(transicoes),
        "motivos_frequentes": dict(motivos.most_common(20)),
        "documentos": resultados,
        "observacao": (
            "A classificação é heurística. Nenhum documento "
            "está aprovado para treinamento. A limpeza "
            "calculada não altera os arquivos originais."
        ),
    }

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== CYGNUS V7 | FILTRO DE QUALIDADE V2 ===")
    print("Documentos analisados:", len(resultados))

    for categoria in ("candidato", "revisao", "rejeitado"):
        print(
            f"{categoria}: {categorias[categoria]} documentos | "
            f"{tokens_categorias[categoria]:,} tokens"
        )

    print("\nTRANSIÇÕES:")
    for transicao, quantidade in sorted(transicoes.items()):
        print(f"  {transicao}: {quantidade}")

    print("\nMOTIVOS MAIS FREQUENTES:")
    for motivo, quantidade in motivos.most_common(10):
        print(f"  {motivo}: {quantidade}")

    alterados = [
        item for item in resultados
        if item["categoria_v1"] != item["categoria_v2"]
    ]

    print("\nDOCUMENTOS QUE MUDARAM DE CATEGORIA:", len(alterados))

    for item in alterados[:args.amostras]:
        texto = (
            ORIGEM / item["arquivo"]
        ).read_text(encoding="utf-8")

        trecho = re.sub(r"\s+", " ", texto[:300])

        print("\nArquivo:", item["arquivo"])
        print(
            "Mudança:",
            item["categoria_v1"],
            "->",
            item["categoria_v2"],
        )
        print("Motivos:", item["motivos_v2"])
        print("Trecho:", trecho)

    print("\nRelatório:", SAIDA)
    print("FILTRAGEM V2 CONCLUÍDA")


if __name__ == "__main__":
    main()
