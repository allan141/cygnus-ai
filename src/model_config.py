
# Cygnus AI - Fase 15
# Configuracoes de arquitetura

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfiguracaoCygnus:
    dimensao_embedding: int = 128
    numero_cabecas: int = 4
    numero_camadas: int = 4
    tamanho_contexto: int = 128

    def validar(self):
        if self.dimensao_embedding <= 0:
            raise ValueError(
                "Embedding deve ser positivo"
            )

        if self.numero_cabecas <= 0:
            raise ValueError(
                "Numero de cabecas deve ser positivo"
            )

        if self.numero_camadas <= 0:
            raise ValueError(
                "Numero de camadas deve ser positivo"
            )

        if self.tamanho_contexto <= 0:
            raise ValueError(
                "Contexto deve ser positivo"
            )

        if (
            self.dimensao_embedding
            % self.numero_cabecas != 0
        ):
            raise ValueError(
                "Embedding deve ser divisivel "
                "pelo numero de cabecas"
            )

    def para_modelo(self, tamanho_vocabulario):
        self.validar()

        if tamanho_vocabulario <= 0:
            raise ValueError(
                "Vocabulario deve ser positivo"
            )

        return {
            "tamanho_vocabulario": tamanho_vocabulario,
            "dimensao_embedding": self.dimensao_embedding,
            "numero_cabecas": self.numero_cabecas,
            "numero_camadas": self.numero_camadas,
            "tamanho_contexto": self.tamanho_contexto
        }
