# Pesquisa CRIEC

Repositório voltado ao processamento e automação de dados GOES-19 disponibilizados pelo INPE para uso em atividades de pesquisa.

## Objetivo

Este projeto organiza um pipeline em Python para:

- baixar dados GOES-19 em formato NetCDF;
- recortar áreas de interesse geográfica;
- gerar arquivos de saída em NetCDF e PNG para análise posterior.

## Estrutura do projeto

- [Automações/goes_toolkit.py](Automações/goes_toolkit.py): módulo principal com as funções de download, recorte e geração de imagens.
- [requirements.txt](requirements.txt): dependências Python do projeto.
- [pyproject.toml](pyproject.toml): configuração do projeto e dependências.

## Requisitos

- Python 3.10 ou superior
- Dependências listadas em [requirements.txt](requirements.txt)

## Observações

- A pasta [.venv](.venv) não é versionada.
- O projeto já está configurado para ignorar arquivos temporários, cache Python e a virtual environment.
