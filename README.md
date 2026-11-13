# Cygnus AI

**Uma inteligência artificial desenvolvida do zero, do Transformer à integração com ferramentas.**

> **Status:** projeto experimental em desenvolvimento · **Marco atual:** v0.4.0 (Cygnus v4)

O **Cygnus AI** é um projeto independente de pesquisa e engenharia de inteligência artificial, criado para estudar, implementar, treinar e avaliar modelos próprios de linguagem e, progressivamente, transformá-los em um assistente capaz de responder perguntas, consultar informações externas e interpretar diferentes tipos de entrada.

O objetivo **não** é colocar uma interface sobre um LLM comercial ou usar um modelo de linguagem já pré-treinado como núcleo do sistema. A proposta é construir e compreender os componentes fundamentais da IA — desde a tokenização e a arquitetura Transformer até o treinamento, a avaliação e a orquestração de ferramentas.

O projeto utiliza bibliotecas e infraestrutura de desenvolvimento, como **Python e PyTorch**. Futuramente, poderá integrar serviços externos de busca na web, mas **sem delegar a geração de respostas a LLMs prontos**.

## Visão do projeto

A meta de longo prazo é criar um assistente que consiga:

- Compreender perguntas e produzir respostas úteis em português.
- Desenvolver conhecimento especializado em **inteligência artificial, programação, estética e cosmética**.
- Consultar a internet por meio de uma **Search API** quando a informação for externa, atual ou não estiver disponível localmente.
- Recuperar informações de documentos e apresentar as fontes utilizadas.
- Interpretar imagens, capturas de tela e gráficos por meio de componentes próprios de visão computacional.
- Utilizar ferramentas de forma controlada e indicar incerteza quando não houver evidências suficientes.

**Essas são metas de desenvolvimento, não funcionalidades já disponíveis.** A versão atual é um modelo experimental de geração de texto, ainda incapaz de responder perguntas com confiabilidade.

## O que já foi desenvolvido

O Cygnus já conta com uma base de experimentação de modelos de linguagem implementada em PyTorch:

- **Transformer decoder-only**, com embeddings de tokens e posições, atenção multi-head, blocos Transformer e projeção para o vocabulário.
- **Tokenizador Byte-Level BPE** próprio, com treinamento, codificação, decodificação e persistência.
- **Pipeline de coleta e preparação de corpus**, incluindo textos da Wikipédia, divisão em treinos/validação e preparação de trechos.
- **Treinamento e avaliação**, com AdamW, cálculo de perda, perplexidade, checkpoints e retomada de treinamento.
- **Geração de texto**, com controle de temperatura, top-k e tamanho da saída.
- **Testes automatizados** para componentes do pipeline e do treinamento.

### Experimento Cygnus v4

| Característica | Configuração |
| --- | --- |
| Arquitetura | Transformer decoder-only |
| Parâmetros | **1.350.400** |
| Vocabulário | **2.048 tokens** (Byte-Level BPE) |
| Contexto | **256 tokens** |
| Embeddings | **128 dimensões** |
| Camadas Transformer | **4** |
| Cabeças de atenção | **4** |
| Passos de treinamento | **5.000** |
| Corpus | Artigos em português preparados para treino e validação |

**Resultados registrados:**

| Passo | Perda de validação | Perplexidade |
| ---: | ---: | ---: |
| 1 | 7,8073 | 2.458,45 |
| 1.000 | 6,1193 | 454,54 |
| 2.000 | 5,4332 | 228,88 |
| 3.000 | 5,1618 | 174,47 |
| 4.000 | 5,0293 | 152,83 |
| **4.800** | **4,9525** | **141,53** |
| 5.000 | 4,9601 | 142,60 |

A melhor perda de validação observada ocorreu no **passo 4.800**. A redução da perplexidade demonstra aprendizado de padrões estatísticos do corpus, **não** comprova compreensão semântica ou qualidade conversacional. As gerações atuais ainda apresentam palavras inexistentes, erros gramaticais e incoerências.

> **Nota sobre avaliação:** os resultados foram obtidos em uma configuração experimental com **16 janelas fixas de validação**. Não devem ser interpretados como benchmark amplo de capacidade linguística nem comparados diretamente com perplexidades de tokenizadores diferentes.

## Arquitetura atual e evolução planejada

```text
                   CYGNUS AI — VISÃO DE LONGO PRAZO

                        Pergunta do usuário
                                |
                       Interface / API
                                |
                      Orquestrador próprio
                                |
              +-----------------+-----------------+
              |                 |                 |
       Modelo de linguagem   Busca e RAG      Visão computacional
       próprio (Transformer) Search API       imagens / prints
              |                 |                 |
              +-----------------+-----------------+
                                |
                   Evidências e verificação
                                |
                       Resposta ao usuário
```

**No v4**, a parte implementada é principalmente o **modelo de linguagem e seu pipeline de treinamento e geração**. Os demais módulos representam o caminho planejado para versões futuras.

## Roadmap

- [x] Implementar a base do Transformer decoder-only em PyTorch.
- [x] Desenvolver tokenização Byte-Level BPE.
- [x] Criar coletores e preparação de corpus em português.
- [x] Implementar treinamento, checkpoints, retomada e métricas.
- [x] Treinar e avaliar o Cygnus v4 por 5.000 passos.
- [x] Criar uma interface inicial de geração por terminal.
- [ ] **Cygnus v5:** ampliar, limpar e equilibrar o corpus em português.
- [ ] Melhorar a geração de palavras, frases e pequenos textos coerentes.
- [ ] Criar datasets e avaliações de perguntas e respostas por domínio.
- [ ] Desenvolver recuperação local de informações (por exemplo, BM25/TF-IDF).
- [ ] Integrar uma Search API e leitura de páginas para consultas externas.
- [ ] Criar respostas fundamentadas em evidências, com fontes e tratamento de incerteza.
- [ ] Desenvolver módulos próprios para leitura de imagens, prints e gráficos.
- [ ] Integrar os componentes em um assistente com API e interface de chat.
- [ ] Medir acurácia, qualidade das respostas, robustez e desempenho.
- [ ] Publicar documentação técnica, experimentos e demonstrações reproduzíveis.

## Organização do repositório

Os diretórios e módulos mais relevantes incluem:

```text
cygnus-ai/
├── src/
│   ├── language_model.py          # Modelo Transformer
│   ├── byte_bpe_tokenizer.py      # Tokenizador próprio
│   ├── model_config.py            # Configuração da arquitetura
│   ├── corpus_v4.py               # Preparação do corpus v4
│   ├── tokenizer_v4_benchmark.py  # Comparação de tokenizadores
│   ├── v4_trainer.py              # Treinamento do v4
│   ├── cygnus_v4_chat.py          # Geração de texto
│   └── ...                        # Coletores e ferramentas auxiliares
├── tests/                         # Testes automatizados
├── data/                          # Dados locais (parte ignorada pelo Git)
└── checkpoints/                   # Pesos e estados locais (ignorados pelo Git)
```

## Executando localmente

**Pré-requisitos:** Python, ambiente virtual e dependências do projeto (incluindo PyTorch e pytest). Execute os comandos a partir da raiz do repositório, com as dependências instaladas.

```powershell
# Ativar ambiente virtual no Windows / PowerShell
.\.venv\Scripts\Activate.ps1

# Executar testes
python -m pytest tests/ -q

# Gerar texto com o checkpoint v4 local
python -m src.cygnus_v4_chat --max-tokens 80 --temperatura 0.7 --top-k 20

# Gerar texto a partir de um prompt
python -m src.cygnus_v4_chat --prompt "A inteligência artificial" --max-tokens 80 --temperatura 0.7 --top-k 20
```

O gerador requer os **arquivos de checkpoint e tokenizador compatíveis**, produzidos no treinamento. Esses arquivos não são distribuídos neste repositório porque `checkpoints/` está excluído do versionamento. Portanto, clonar o código, por si só, **não basta para executar a geração com o modelo treinado**.

Exemplo de treinamento/retomada, após preparar o corpus e os artefatos necessários:

```powershell
python -m src.v4_trainer --passos 5000 --batch-size 2 --avaliar-cada 100 --retomar
```

Esse comando é uma referência do experimento realizado, não um procedimento de instalação completa a partir de um clone limpo.

## Princípios de engenharia

O desenvolvimento do Cygnus prioriza:

1. **Autoria técnica:** implementar e compreender os componentes centrais, sem usar LLMs pré-treinados como motor de respostas.
2. **Experimentos reproduzíveis:** registrar configurações, dados, checkpoints e métricas.
3. **Avaliação honesta:** distinguir melhoria da perda de validação de capacidade real de responder corretamente.
4. **Modularidade:** separar modelo de linguagem, recuperação, visão e ferramentas.
5. **Evolução incremental:** validar cada capacidade antes de ampliar a arquitetura.
6. **Uso responsável de dados:** documentar origens, licenças e restrições dos conjuntos utilizados.

## Tecnologias

**Atuais:** Python, PyTorch, Byte-Level BPE, Transformer decoder-only, pytest e Git/GitHub.

**Planejadas:** FastAPI, mecanismos de recuperação de informação, Search APIs, processamento de documentos e visão computacional.

## Motivação

O Cygnus AI também é um projeto de portfólio de **Engenharia de Inteligência Artificial**. Seu propósito é demonstrar competências práticas em arquitetura de modelos, treinamento, engenharia de dados, avaliação, experimentação e integração de sistemas — incluindo a análise transparente dos desafios e limitações encontrados durante o desenvolvimento.

---

**Cygnus AI** — Construindo inteligência artificial, componente por componente.
