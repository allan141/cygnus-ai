
import pytest
import torch

from src.model_config import ConfiguracaoCygnus
from src.model_analyzer import (
    analisar_modelo,
    testar_forward as executar_forward
)


def test_configuracao_padrao():
    config = ConfiguracaoCygnus()

    assert config.dimensao_embedding == 128
    assert config.numero_camadas == 4
    assert config.tamanho_contexto == 128


def test_configuracao_valida():
    config = ConfiguracaoCygnus()
    config.validar()


def test_embedding_incompativel():
    config = ConfiguracaoCygnus(
        dimensao_embedding=130,
        numero_cabecas=4
    )

    with pytest.raises(ValueError):
        config.validar()


def test_camadas_invalidas():
    config = ConfiguracaoCygnus(
        numero_camadas=0
    )

    with pytest.raises(ValueError):
        config.validar()


def test_contexto_invalido():
    config = ConfiguracaoCygnus(
        tamanho_contexto=0
    )

    with pytest.raises(ValueError):
        config.validar()


def test_vocabulario_invalido():
    config = ConfiguracaoCygnus()

    with pytest.raises(ValueError):
        config.para_modelo(0)


def test_criacao_modelo_v3():
    resultado = analisar_modelo(
        tamanho_vocabulario=512
    )

    assert resultado["parametros"] > 0
    assert resultado["treinaveis"] > 0


def test_forward_modelo_v3():
    resultado = analisar_modelo(
        tamanho_vocabulario=512
    )

    formato = executar_forward(
        resultado["modelo"]
    )

    assert formato == (1, 16, 512)


def test_memoria_positiva():
    resultado = analisar_modelo(
        tamanho_vocabulario=512
    )

    assert resultado["memoria_pesos_mb"] > 0


def test_parametros_treinaveis():
    resultado = analisar_modelo(
        tamanho_vocabulario=512
    )

    assert (
        resultado["parametros"]
        == resultado["treinaveis"]
    )
