# --------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------

import os
import time
import logging
from datetime import datetime, timedelta, timezone
import requests
from pystac_client import Client
import shutil
import tempfile
import uuid
import concurrent.futures
import threading

import numpy as np
import xarray as xr
from scipy.interpolate import griddata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Config globais
# --------------------------------------------------------------------------

logger = logging.getLogger(__name__)

_NETCDF_LOCK = threading.Lock()

# --------------------------------------------------------------------------
# Utilidades gerais
# --------------------------------------------------------------------------

def montar_colecao_stac(
    serie: str
) -> str:
    """Monta o identificador da coleção STAC a partir da série do satélite GOES.

    Args:
        serie (str): Série do satélite (ex: 'GOES16', 'GOES19').

    Returns:
        str: Identificador da coleção no catálogo STAC do INPE (ex: 'GOES19-L2-CMI-1').

    Note:
        Confirme o identificador exato e a extensão temporal real da coleção
        em https://data.inpe.br/bdc/stac/v1/collections antes. 
        Nem toda série necessariamente segue este padrão de nome, e
        cada coleção só existe durante o período em que aquele satélite
        específico esteve operacional (ex: 'GOES19-L2-CMI-1' só tem dados a
        partir de 02/04/2025, quando o GOES-19 assumiu como GOES-East; dados
        anteriores estão em 'GOES16-L2-CMI-1').
    """
    return f"{serie}-L2-CMI-1"

def _dividir_intervalo_em_blocos(
    datetime_str: str, 
    dias_por_bloco: int
) -> list[str]:
    """Divide um intervalo STAC 'inicio/fim' (ISO 8601) em blocos consecutivos
    de até `dias_por_bloco` dias, retornando uma lista de novos intervalos
    'inicio/fim' no mesmo formato.

    Se `datetime_str` não for um intervalo (não contém '/'), ou já cobrir
    `dias_por_bloco` dias ou menos, retorna [datetime_str] sem dividir.

    Args:
        datetime_str (str): Intervalo ISO 8601 'inicio/fim' (ex:
            '2020-01-01T00:00:00Z/2026-01-01T00:00:00Z') ou uma data única.
        dias_por_bloco (int): Tamanho máximo de cada bloco, em dias.

    Returns:
        list[str]: Lista de intervalos 'inicio/fim', cobrindo o intervalo
            original sem sobreposição.
    """
    if "/" not in datetime_str:
        return [datetime_str]

    inicio_str, fim_str = datetime_str.split("/", 1)
    inicio = datetime.fromisoformat(inicio_str.replace("Z", "+00:00"))
    fim = datetime.fromisoformat(fim_str.replace("Z", "+00:00"))

    if fim <= inicio:
        return [datetime_str]

    blocos = []
    atual = inicio
    delta = timedelta(days=dias_por_bloco)

    while atual < fim:
        proximo = min(atual + delta, fim)
        blocos.append(
            f"{atual.strftime('%Y-%m-%dT%H:%M:%SZ')}/{proximo.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        )
        atual = proximo

    return blocos

def _filtrar_por_intervalo(
    items: list,
    intervalo_minutos: float | None,
    tolerancia_minutos: float = 5.0,
) -> list:
    """Reduz a lista de itens STAC a uma imagem a cada `intervalo_minutos`.

    O GOES gera um full disk a cada ~10 min. Para montar uma base mais leve
    (ex: de hora em hora), a função considera como horários-alvo os múltiplos de
    `intervalo_minutos` contados a partir de 00:00 UTC e, para cada um, escolhe o
    item mais próximo, desde que esteja a no máximo `tolerancia_minutos` dele.
    Assim a seleção não depende de o timestamp cair exatamente em HH:00:00, e
    todas as bandas processadas a partir da mesma lista de itens ficam com os
    mesmos instantes.

    Se a imagem de um horário-alvo não existir, o horário fica sem dado (lacuna)
    em vez de ser preenchido com uma imagem de outro instante, e um aviso é
    registrado. Se nenhum item chegar perto de nenhum horário-alvo (por exemplo,
    um intervalo que não casa com o horário das imagens), usa-se o item mais
    próximo de cada horário, com aviso, para não devolver uma lista vazia.

    Args:
        items (list): Itens STAC (pystac.Item) retornados pela busca.
        intervalo_minutos (float, optional): Intervalo desejado entre imagens, em
            minutos. None mantém todos os itens. Valores que dividem 1440 (10, 20,
            30, 60, 120, 180, 360, 720, 1440) dão horários-alvo iguais todos os dias;
            valores menores ou iguais à cadência nativa (10 min) não removem nada.
        tolerancia_minutos (float, optional): Distância máxima, em minutos, entre o
            item e o horário-alvo. Padrão é 5 (metade da cadência nativa).

    Returns:
        list: Itens escolhidos, em ordem cronológica.

    Raises:
        ValueError: Se `intervalo_minutos` não for positivo.
    """
    if intervalo_minutos is None:
        return items

    seg = float(intervalo_minutos) * 60.0
    tol = float(tolerancia_minutos) * 60.0

    dentro = {}     # horário-alvo -> (desvio em segundos, item)
    qualquer = {}   # idem, sem tolerância (usado só como último recurso)
    sem_data = []

    for item in items:
        dt = item.datetime
        if dt is None:
            dt = item.common_metadata.start_datetime

        if dt is None:
            sem_data.append(item)
            continue

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        ts = dt.timestamp()
        alvo = int(np.floor(ts / seg + 0.5))
        desvio = abs(ts - alvo * seg)

        atual = qualquer.get(alvo)
        if atual is None or desvio < atual[0]:
            qualquer[alvo] = (desvio, item)

        if desvio <= tol:
            atual = dentro.get(alvo)
            if atual is None or desvio < atual[0]:
                dentro[alvo] = (desvio, item)

    escolhas = dentro
    if not dentro and qualquer:
        logger.warning(
            f"Nenhum item está a menos de {tolerancia_minutos:g} min de um horário-alvo de "
            f"{intervalo_minutos:g} min. Usando o item mais próximo de cada horário."
        )
        escolhas = qualquer

    escolhidos = [escolhas[k][1] for k in sorted(escolhas)]

    if len(escolhas) > 1 and intervalo_minutos >= 10:
        chaves = sorted(escolhas)
        faltando = (chaves[-1] - chaves[0] + 1) - len(chaves)
        if faltando > 0:
            logger.warning(f"{faltando} horário(s)-alvo sem imagem no período (lacunas nos dados).")

    if sem_data:
        logger.warning(f"{len(sem_data)} item(ns) sem data não puderam ser filtrados por intervalo e foram mantidos.")
        escolhidos.extend(sem_data)

    logger.info(f"Intervalo de {intervalo_minutos:g} min: {len(escolhidos)} de {len(items)} itens mantidos.")

    return escolhidos

def _executar_com_retry(
    func,
    *args,
    max_tentativas: int = 3,
    espera_inicial: float = 2.0,
    excecoes=(Exception,),
    **kwargs,
):
    """Executa `func(*args, **kwargs)` com retentativas e backoff exponencial.

    Usado para operações de rede (download HTTP, consulta STAC) sujeitas a
    falhas transitórias (timeout, reset de conexão, instabilidade momentânea
    do serviço).

    Args:
        func (Callable): Função a executar.
        max_tentativas (int, optional): Número máximo de tentativas. Padrão é 3.
        espera_inicial (float, optional): Espera (segundos) antes da 2ª tentativa;
            dobra a cada nova tentativa. Padrão é 2.0.
        excecoes (tuple, optional): Exceções que disparam nova tentativa.
            Padrão é (Exception,).

    Returns:
        O retorno de `func`, se alguma tentativa tiver sucesso.

    Raises:
        A última exceção capturada, se todas as tentativas falharem.
    """
    ultima_excecao = None
    espera = espera_inicial

    for tentativa in range(1, max_tentativas + 1):
        try:
            return func(*args, **kwargs)

        except excecoes as e:
            ultima_excecao = e

            if tentativa == max_tentativas:
                break

            logger.warning(
                f"Tentativa {tentativa}/{max_tentativas} falhou ({e}). "
                f"Nova tentativa em {espera:.1f}s..."
            )
            time.sleep(espera)
            espera *= 2

    raise ultima_excecao

# --------------------------------------------------------------------------
# Funções da pipeline
# --------------------------------------------------------------------------

def regradear_para_latlon(
    ds_path: str,
    lat_min: float | None = None,
    lat_max: float | None = None,
    lon_min: float | None = None,
    lon_max: float | None = None,
    reg_value: float = 0.05,
    reg_method: str = "linear",
    output_dir: str = "dados_goes_roi",
    arq_name: str | None = None
) -> xr.Dataset:
    """Regradeia o CMI de um arquivo NetCDF do GOES (grade nativa x/y, ângulo de
    varredura) para uma grade REGULAR em latitude/longitude, cobrindo
    exatamente o retângulo (lat_min, lat_max, lon_min, lon_max) informado —
    o mesmo ROI passado para `baixar_roi_goes`/`baixar_roi_multibanda_goes`/`converter_latlon_para_xy`.

    Usa `converter_xy_para_latlon` para obter latitude/longitude por pixel
    e reamostra o CMI como dado disperso (scattered) via `scipy.interpolate.griddata`

    Args:
        ds_path (str): Caminho do NetCDF recortado (ex: saída de `baixar_roi_goes`).
        lat_min (float, optional): Latitude mínima (Sul) do retângulo de saída, em
            graus. Se None, usa a ROI gravada nos atributos globais do próprio
            arquivo (ver `baixar_roi_goes`).
        lat_max (float, optional): Latitude máxima (Norte). Mesma regra de lat_min.
        lon_min (float, optional): Longitude mínima (Oeste). Mesma regra de lat_min.
        lon_max (float, optional): Longitude máxima (Leste). Mesma regra de lat_min.
        reg_value (float, optional): Espaçamento da grade de saída, em graus.
            Padrão é 0.05.
        reg_method (str, optional): Método de interpolação do `griddata`:
            'nearest', 'linear' ou 'cubic'. Padrão é 'linear'.
        output_dir (str, optional): Diretório de saída. Padrão é 'dados_goes_roi'.
        arq_name (str, optional): Nome do arquivo de saída. Se None, é derivado
            do nome do arquivo de entrada (ex: 'crop_X_B14.nc' -> 'reg_X_B14.nc'),
            evitando que múltiplas chamadas em lote se sobrescrevam por
            usarem sempre o mesmo nome padrão.

    Returns:
        xr.Dataset: Dataset com 'CMI' regradeado, dimensões ('lat', 'lon').

    Raises:
        ValueError: Se a ROI não for informada e também não existir nos
            atributos globais do arquivo de entrada.

    Example:
        >>> ds_reg = regradear_para_latlon(
        ...     ds_path="dados_goes_roi/B14/crop_GOES19_CMI_..._B14.nc",
        ... )  # usa a ROI gravada no próprio arquivo
        >>> print(ds_reg)
    """

    with _NETCDF_LOCK, xr.open_dataset(ds_path, engine='netcdf4') as ds:
        roi_do_arquivo = {
            "lat_min": ds.attrs.get("roi_lat_min"),
            "lat_max": ds.attrs.get("roi_lat_max"),
            "lon_min": ds.attrs.get("roi_lon_min"),
            "lon_max": ds.attrs.get("roi_lon_max"),
        }
        ds_geo = converter_xy_para_latlon(ds).load()

    # Se o chamador não informou a ROI explicitamente, usa a ROI gravada no
    # próprio arquivo de recorte. Isso elimina a necessidade de repetir os
    # mesmos lat/lon min/max nas duas etapas (recorte + regrid) - e o risco
    # de as duas chamadas ficarem dessincronizadas por engano.
    lat_min = lat_min if lat_min is not None else roi_do_arquivo["lat_min"]
    lat_max = lat_max if lat_max is not None else roi_do_arquivo["lat_max"]
    lon_min = lon_min if lon_min is not None else roi_do_arquivo["lon_min"]
    lon_max = lon_max if lon_max is not None else roi_do_arquivo["lon_max"]

    if None in (lat_min, lat_max, lon_min, lon_max):
        raise ValueError(
            f"ROI não informada e não encontrada nos metadados de '{ds_path}'. "
            "Passe lat_min/lat_max/lon_min/lon_max explicitamente, ou recorte "
            "o arquivo com uma versão da pipeline que grava a ROI nos "
            "atributos globais (ver baixar_roi_goes)."
        )

    lat2d = ds_geo["latitude"].values    # (y, x)
    lon2d = ds_geo["longitude"].values   # (y, x)
    cmi = ds_geo["CMI"].values           # (y, x)

    lat_lo, lat_hi = min(lat_min, lat_max), max(lat_min, lat_max)
    lon_lo, lon_hi = min(lon_min, lon_max), max(lon_min, lon_max)

    # Grade de saída: exatamente o retângulo do ROI, não os limites do dado
    # recortado (que ficam distorcidos/tortos por causa da projeção).
    reg_lon = np.arange(lon_lo, lon_hi, reg_value)
    reg_lat = np.arange(lat_hi, lat_lo, -reg_value)
    lon_grid, lat_grid = np.meshgrid(reg_lon, reg_lat)

    # Remove pixels "de espaço" (NaN vindos da navegação, fora do disco terrestre)
    valid = np.isfinite(lat2d) & np.isfinite(lon2d) & np.isfinite(cmi)

    cmi_reg = griddata(
        points=(lon2d[valid], lat2d[valid]),
        values=cmi[valid],
        xi=(lon_grid, lat_grid),
        method=reg_method,
    )

    buracos = np.isnan(cmi_reg)

    # 'linear'/'cubic' não extrapolam pontos, fora do casco convexo dos dados
    # de entrada saem como NaN. Preenche só esses buracos residuais
    # com 'nearest'.
    if reg_method != "nearest" and np.any(buracos):
        cmi_nearest = griddata(
            points=(lon2d[valid], lat2d[valid]),
            values=cmi[valid],
            xi=(lon_grid, lat_grid),
            method="nearest",
        )
        cmi_reg = np.where(buracos, cmi_nearest, cmi_reg)

    ds_reg = xr.Dataset(
        {"CMI": (("lat", "lon"), cmi_reg.astype("float32"))},
        coords={
            "lat": reg_lat.astype("float32"),
            "lon": reg_lon.astype("float32"),
        },
    )

    ds_reg["lat"].attrs.update(units="degrees_north", standard_name="latitude", axis="Y")
    ds_reg["lon"].attrs.update(units="degrees_east", standard_name="longitude", axis="X")

    if arq_name is None:
        base = os.path.basename(ds_path)
        if base.startswith("crop_"):
            base = base[len("crop_"):]
        arq_name = f"reg_{base}"

    regridded_dir = os.path.join(output_dir, "Regridded_Data")
    os.makedirs(regridded_dir, exist_ok=True)
    destino = os.path.join(regridded_dir, arq_name)
    with _NETCDF_LOCK:
        ds_reg.to_netcdf(destino)

    logger.info(f"Arquivo regradeado salvo em: {destino}")

    return ds_reg

def gerar_png_do_dataset(
    ds, 
    path_nc: str
) -> str:
    """Gera um PNG a partir de um dataset xarray já carregado na memória.

    Procura a variável 'CMI' no dataset; se não existir, usa a primeira
    variável de dados disponível. O PNG é salvo no mesmo caminho do NetCDF
    de origem, trocando a extensão '.nc' por '.png'.

    Args:
        ds (xr.Dataset): Dataset já carregado na memória, contendo a
            variável de imagem a ser plotada.
        path_nc (str): Caminho do arquivo NetCDF de origem, usado para
            derivar o nome do PNG de saída (mesmo caminho, extensão '.png').

    Returns:
        str: Caminho local do arquivo PNG gerado.

    Raises:
        ValueError: Se nenhuma variável de dados for encontrada no dataset.
    """

    # Procura a variavel contendo a imagem
    if "CMI" in ds.data_vars:
        var_name = "CMI"

    else:
        var_name = next(iter(ds.data_vars), None)

    if var_name is None:
        raise ValueError(f"Nenhuma variável de imagem encontrada em {path_nc}")

    # Plota usando o Matplot
    plt.figure(figsize=(8, 8))
    plt.imshow(ds[var_name])
    png_path = path_nc.replace('.nc', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()

    return png_path

def buscar_itens_goes(
    datetime_str: str,
    collection: str = "GOES19-L2-CMI-1",
    intervalo_minutos: float | None = None,
) -> list:
    """Consulta a STAC API do INPE uma única vez e retorna os itens encontrados.

    O resultado da busca depende apenas do intervalo de tempo (e da coleção),
    não da banda espectral. Centralizar a busca aqui permite reaproveitar os
    mesmos itens ao processar várias bandas (ver `baixar_roi_multibanda_goes`), evitando
    repetir a mesma consulta à API STAC uma vez por banda.

    A conexão e a busca em si são tentadas com retentativas automáticas (ver
    `_executar_com_retry`), já que são operações de rede sujeitas a falhas
    transitórias.

    Args:
        datetime_str (str): Data/hora única em ISO 8601 (AAAA-MM-DDTHH:MM:SSZ) ou intervalo temporal.
        collection (str, optional): Coleção STAC a consultar. Padrão é 'GOES19-L2-CMI-1'.
        intervalo_minutos (float, optional): Intervalo entre as imagens mantidas, em
            minutos (ver `_filtrar_por_intervalo`). Ex: 60 para uma imagem por hora.
            Padrão é None (todas as imagens, a cada ~10 min). O filtro é aplicado
            depois da busca: a consulta ao catálogo é a mesma, mas só os itens
            escolhidos seguem para o download.

    Returns:
        list: Itens (pystac.Item) encontrados no catálogo.
    """

    # Consulta a API do INPE para criar uma stack
    api_url = "https://data.inpe.br/bdc/stac/v1/"
    logger.info(f"Conectando à STAC API: {api_url}")

    def _buscar():
        client = Client.open(api_url)
        logger.info(f"Buscando dados para a coleção '{collection}'...")
        search = client.search(collections=[collection], datetime=datetime_str)
        return list(search.items())

    items = _executar_com_retry(_buscar, max_tentativas=3, excecoes=(Exception,))

    logger.info(f"Total de registros encontrados: {len(items)}")

    items = _filtrar_por_intervalo(items, intervalo_minutos)

    return items

def baixar_banda_goes(
    banda: str,
    datetime_str: str,
    output_dir: str = "dados_goes",
    items: list | None = None,
    collection: str = "GOES19-L2-CMI-1",
    intervalo_minutos: float | None = None,
) -> list[str]:
    """Busca e realiza o download dos dados em NetCDF do GOES-19 na STAC API do INPE.

    A função consulta o catálogo STAC do Brazil Data Cube (BDC/INPE) para a 
    coleção de dados de reflectância e temperatura do GOES-R (`GOESR-L2-CMI-1`), 
    filtra pelo intervalo temporal fornecido e faz o download direto da banda 
    espectral solicitada.

    Args:
        banda (str): Identificador do asset/banda desejada no item STAC 
            (ex: 'B01' a 'B16' para o sensor ABI do GOES).
        datetime_str (str): Data/hora única em ISO 8601 (AAAA-MM-DDTHH:MM:SSZ) ou intervalo temporal.
            Exemplos: '2026-06-01T14:30:00Z' ou '2026-06-01T14:00:00Z/2026-06-01T15:00:00Z'.
        output_dir (str, optional): Diretório local onde os arquivos NetCDF (.nc) 
            serão armazenados. Criado automaticamente caso não exista. Padrão é 'dados_goes'.
        items (list, optional): Itens STAC já resolvidos (de `buscar_itens_goes`). Se
            None, a função faz a própria busca. Passar itens prontos evita repetir a
            consulta à API quando várias bandas do mesmo intervalo são processadas.
        collection (str, optional): Coleção STAC a consultar, usada apenas quando
            `items` não é informado. Padrão é 'GOES19-L2-CMI-1'.
        intervalo_minutos (float, optional): Intervalo entre as imagens baixadas, em
            minutos (ex: 60 para uma por hora). Usado apenas quando `items` não é
            informado. Padrão é None (todas).

    Returns:
        list[str]: Lista contendo os caminhos locais de todos os arquivos baixados 
            ou já existentes no disco.

    Raises:
        requests.exceptions.RequestException: Em caso de erro de conexão ao realizar o download.
        Exception: Em caso de falha de conexão ou erro no cliente STAC.

    Example:
        >>> baixados = baixar_banda_goes(
        ...     banda="B14",
        ...     datetime_str="2026-06-01T14:30:00Z/2026-06-01T14:35:00Z",
        ...     output_dir="./dados_goes"
        ... )
        >>> print(baixados)
        ['./dados_goes/GOES19_CMI_..._B14.nc']
    """

    # Verifica a existência do diretório de saída
    os.makedirs(output_dir, exist_ok=True)

    # Verifica a existência de uma stack, se não houver cria usando buscar_itens_goes
    if items is None:
        items = buscar_itens_goes(datetime_str, collection=collection, intervalo_minutos=intervalo_minutos)

    # Verifica se a API encontrou algum DataSet
    if len(items) == 0:
        logger.info("Nenhuma imagem encontrada para os parâmetros informados.")
        return []

    ids_baixados = set()
    arquivos_salvos = []

    # Baixa os arquivos e trata exceptions
    for item in items:
        if item.id in ids_baixados:
            continue
            
        logger.info(f"Processando Item ID: {item.id} (Data: {item.datetime})")
        
        if banda in item.assets:
            asset = item.assets[banda]
            file_url = asset.href
            
            file_name = f"{item.id}_{banda}.nc"
            file_path = os.path.abspath(os.path.join(output_dir, file_name))
            
            if os.path.exists(file_path):
                logger.info(f"Arquivo já existe localmente: {file_name}. Pulando download.")
                ids_baixados.add(item.id)
                arquivos_salvos.append(file_path)
                continue
                
            logger.info(f"Baixando de: {file_url}")
            logger.info(f"Salvando em: {file_path}...")

            try:
                response = _executar_com_retry(
                    requests.get,
                    file_url,
                    stream=True,
                    timeout=(10, 300),
                    max_tentativas=3,
                    excecoes=(requests.exceptions.ConnectionError, requests.exceptions.Timeout),
                )
            except requests.exceptions.RequestException as e:
                logger.error(f"Falha ao baixar {file_url} após múltiplas tentativas: {e}")
                continue

            if response.status_code == 200:

                # Garante que o diretório existe antes de escrever
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

                # Confirma que o arquivo foi escrito no disco 
                if os.path.exists(file_path):
                    logger.info(f"Download concluído com sucesso: {file_name}")
                    ids_baixados.add(item.id)
                    arquivos_salvos.append(file_path)
                else:
                    logger.error(f"Falha ao salvar o arquivo: {file_path}")
            else:
                logger.error(f"Erro ao baixar o arquivo. Status HTTP: {response.status_code}")
        else:
            logger.warning(f"A banda '{banda}' não foi encontrada neste item.")

    return arquivos_salvos

def _angulo_central_graus(
    lat_graus: np.ndarray,
    lon_graus: np.ndarray,
    lon_origem_graus: float,
    lat_origem_graus: float = 0.0,
) -> np.ndarray:
    """Ângulo central geocêntrico (graus) entre o ponto sub-satélite
    (lat_origem_graus, lon_origem_graus) e cada ponto (lat_graus, lon_graus),
    via lei dos cossenos esférica.
    """
    lat1 = np.radians(lat_origem_graus)
    lon1 = np.radians(lon_origem_graus)
    lat2 = np.radians(lat_graus)
    lon2 = np.radians(lon_graus)

    cos_ang = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2)

    return np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))

def _projetar_latlon_para_scan_angle(
        lat_graus, 
        lon_graus, 
        r_eq, 
        r_pol, 
        h_dist, 
        lambda_0_rad
    ):
    """Converte lat/lon (graus) em ângulos de varredura (x, y) do GOES-R, em
    radianos. Núcleo escalar/vetorial usado por `converter_latlon_para_xy`
    (que amostra as bordas de uma caixa) e reutilizável em testes
    automatizados (round-trip com `converter_xy_para_latlon`).

    Fórmulas de GOES-R PUG-L2 (vol 5), seção 4.2.8.

    Args:
        lat_graus, lon_graus: Coordenadas de entrada, em graus (escalar ou array).
        r_eq, r_pol (float): Raios equatorial e polar do elipsoide, em metros.
        h_dist (float): Distância do centro da Terra ao satélite, em metros.
        lambda_0_rad (float): Longitude do ponto sub-satélite, em radianos.

    Returns:
        tuple[np.ndarray, np.ndarray]: (x_angle, y_angle) em radianos.
    """
    phi = np.radians(lat_graus)
    lamb = np.radians(lon_graus)

    e2 = (r_eq**2 - r_pol**2) / (r_eq**2)
    phi_c = np.arctan((r_pol**2 / r_eq**2) * np.tan(phi))
    r_c = r_pol / np.sqrt(1 - e2 * np.cos(phi_c)**2)

    sx = h_dist - r_c * np.cos(phi_c) * np.cos(lamb - lambda_0_rad)
    sy = -r_c * np.cos(phi_c) * np.sin(lamb - lambda_0_rad)
    sz = r_c * np.sin(phi_c)

    x_angle = np.arcsin(-sy / np.sqrt(sx**2 + sy**2 + sz**2))
    y_angle = np.arctan(sz / sx)

    return x_angle, y_angle

def converter_latlon_para_xy(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    margem_graus: float = 0.5,
    n_amostras: int = 25,
    lon_origin_graus: float = -75.2,
) -> tuple[float, float, float, float]:
    """Converte limites geográficos (Lat/Lon) nos limites de varredura (x, y) do GOES-R.

    A partir das coordenadas de uma caixa delimitadora em graus decimais 
    (lat_min, lat_max, lon_min, lon_max), a função utiliza as constantes orbitais 
    do sensor ABI (comuns a toda a série GOES-R: 16/17/18/19 - o que muda entre
    satélites/posições orbitais é a longitude do ponto sub-satélite) para
    calcular os limites correspondentes na Fixed Grid (GEOS) do satélite.

    Como a projeção geoestacionária é não-linear, um retângulo em Lat/Lon NÃO
    corresponde a um retângulo em x/y (vira um quadrilátero distorcido, mais
    forte quanto mais longe do nadir do satélite). Por isso, em vez de converter
    só os 4 cantos da caixa, a função amostra pontos ao longo das 4 bordas
    inteiras e ainda soma uma margem de segurança em graus antes de converter,
    garantindo que o recorte final em x/y cubra o retângulo Lat/Lon inteiro
    (sem faltar pixel nas bordas).

    Também emite um aviso (log) se parte da ROI (incluindo a margem) cair além
    do disco visível pelo satélite naquela posição orbital - nesse caso, o
    arquivo final teria apenas preenchimento (fill-value/NaN) na região
    afetada.

    Args:
        lat_min (float): Latitude mínima (Sul) em graus decimais.
        lat_max (float): Latitude máxima (Norte) em graus decimais.
        lon_min (float): Longitude mínima (Oeste) em graus decimais.
        lon_max (float): Longitude máxima (Leste) em graus decimais.
        margem_graus (float, optional): Margem extra (em graus) somada em torno
            da caixa antes de converter para x/y, absorvendo a distorção da
            projeção nas bordas. Padrão é 0.5.
        n_amostras (int, optional): Número de pontos amostrados ao longo de
            cada uma das 4 bordas do retângulo (não só os cantos). Padrão é 25.
        lon_origin_graus (float, optional): Longitude do ponto sub-satélite, em
            graus decimais. Padrão é -75.2 (posição "GOES-East", usada tanto
            pelo GOES-16 quanto pelo GOES-19). Use -137.2 para a posição
            "GOES-West" (ex: GOES-17/18), se necessário.

    Returns:
        tuple[float, float, float, float]: Tupla contendo os limites em radianos:
            (x_min, x_max, y_min, y_max)

    Example:
        >>> x_min, x_max, y_min, y_max = converter_latlon_para_xy(
        ...     lat_min=-24.0, lat_max=-20.0,
        ...     lon_min=-48.0, lon_max=-43.0
        ... )
        >>> print(f"x: [{x_min:.6f}, {x_max:.6f}], y: [{y_min:.6f}, {y_max:.6f}]")
    """

    R_EQ = 6378137.0          # Raio equatorial em metros (GRS80)
    R_POL = 6356752.31414     # Raio polar em metros
    H_SAT = 35786023.0        # Altura de perspectiva do satélite em metros
    H_DIST = H_SAT + R_EQ     # Distância do centro da Terra ao satélite (m)

    # Normaliza os limites 
    lat_lo, lat_hi = min(lat_min, lat_max), max(lat_min, lat_max)
    lon_lo, lon_hi = min(lon_min, lon_max), max(lon_min, lon_max)

    # Aplica a margem de segurança antes de amostrar a caixa
    lat_min_m = lat_lo - margem_graus
    lat_max_m = lat_hi + margem_graus
    lon_min_m = lon_lo - margem_graus
    lon_max_m = lon_hi + margem_graus

    # Amostra pontos ao longo das 4 bordas da caixa delimitadora
    lats = np.concatenate([
        np.full(n_amostras, lat_min_m),                     # borda Sul
        np.full(n_amostras, lat_max_m),                     # borda Norte
        np.linspace(lat_min_m, lat_max_m, n_amostras),       # borda Oeste
        np.linspace(lat_min_m, lat_max_m, n_amostras),       # borda Leste
    ])
    lons = np.concatenate([
        np.linspace(lon_min_m, lon_max_m, n_amostras),       # borda Sul
        np.linspace(lon_min_m, lon_max_m, n_amostras),       # borda Norte
        np.full(n_amostras, lon_min_m),                      # borda Oeste
        np.full(n_amostras, lon_max_m),                      # borda Leste
    ])

    limite_central_graus = np.degrees(np.arccos(R_EQ / H_DIST))
    angulo_central = _angulo_central_graus(lats, lons, lon_origin_graus)
    if np.any(angulo_central > limite_central_graus):
        logger.warning(
            "Parte da ROI solicitada (lat=[%.2f, %.2f], lon=[%.2f, %.2f], "
            "incluindo margem de %.2f graus) parece cair fora do disco "
            "visível pelo satélite na posição lon_origin=%.1f graus (limite "
            "~%.1f graus do nadir). Os arquivos resultantes podem conter "
            "apenas preenchimento (fill-value/NaN) nessa região.",
            lat_lo, lat_hi, lon_lo, lon_hi, margem_graus,
            lon_origin_graus, limite_central_graus,
        )

    # Converte graus decimais para radianos
    lambda_0 = np.radians(lon_origin_graus)

    x_angles, y_angles = _projetar_latlon_para_scan_angle(lats, lons, R_EQ, R_POL, H_DIST, lambda_0)

    # Extrai os limites mínimos e máximos da região
    x_min, x_max = float(np.min(x_angles)), float(np.max(x_angles))
    y_min, y_max = float(np.min(y_angles)), float(np.max(y_angles))

    return x_min, x_max, y_min, y_max

def converter_xy_para_latlon(
    ds: xr.Dataset,
) -> xr.Dataset:
    """Calcula latitude/longitude por pixel a partir da Fixed Grid (x, y) do GOES.

    Cálculos seguem de GOES-R PUG-L2(vol5), ver seção 4.2.8.1. As constantes de
    projeção (raio equatorial, raio polar, altura e longitude de origem) são
    lidas diretamente dos atributos `goes_imager_projection` do próprio
    arquivo, então esta função já funciona para qualquer satélite da série
    GOES-R sem precisar de ajuste manual. Diferente de x/y, latitude e
    longitude não são separáveis por dimensão (cada pixel depende de x e y ao
    mesmo tempo), por isso são adicionadas como coordenadas auxiliares 2D
    ('latitude', 'longitude'), mantendo x/y como as dimensões do dataset —
    são a entrada esperada pela `regradear_para_latlon`.

    Args:
        ds (xr.Dataset): Dataset já recortado (ex: saída de `baixar_roi_goes`),
            contendo as variáveis 'x', 'y' e 'goes_imager_projection'.

    Returns:
        xr.Dataset: O mesmo dataset, com as coordenadas 2D 'latitude' e
            'longitude' (dims ('y', 'x')) adicionadas.

    Example:
        >>> import xarray as xr
        >>> ds = xr.open_dataset("dados_goes_roi/B14/crop_GOES19_CMI_..._B14.nc")
        >>> ds_geo = converter_xy_para_latlon(ds)
        >>> print(ds_geo["latitude"].shape, ds_geo["longitude"].shape)
    """

    # Arrays de posição em rad(scan angle)
    x = ds['x']
    y = ds['y']

    # Constantes do satélite
    Req = ds['goes_imager_projection'].attrs['semi_major_axis']  # Raio equatorial (m)
    Rpol = ds['goes_imager_projection'].attrs['semi_minor_axis']  #Raio polar (m)
    h = ds['goes_imager_projection'].attrs['perspective_point_height'] # (m)
    H = h + Req
    Lambda_zero = np.radians(ds['goes_imager_projection'].attrs['longitude_of_projection_origin'])  # (deg) -> (rad)

    # Cálculos Intermediários
    a = (np.sin(x))**2 + ((np.cos(x))**2)*((np.cos(y))**2 + (Req**2/Rpol**2)*(np.sin(y))**2)
    b = -2*H*np.cos(x)*np.cos(y)
    c = H**2 - Req**2
    Rs = (-b-np.sqrt(b**2 -4*a*c))/(2*a)
    Sx = Rs*np.cos(x)*np.cos(y)
    Sy = -Rs*np.sin(x)
    Sz = Rs*np.cos(x)*np.sin(y)

    # Conversão
    Phi = np.arctan((Req**2/Rpol**2)*(Sz/np.sqrt((H-Sx)**2 + Sy**2)))  # Latitude geodésica
    Lambda = Lambda_zero - np.arctan(Sy / (H-Sx))  # Longitude geodésica

    # Adiciona lat/lon como coordenadas auxiliares 2D
    DataSet = ds.assign_coords(
        latitude=(('y', 'x'), np.degrees(Phi).transpose('y', 'x').data),
        longitude=(('y', 'x'), np.degrees(Lambda).transpose('y', 'x').data),
    )

    return DataSet

def baixar_roi_goes(
    banda: str,
    datetime_str: str,
    lat_min: float = -20.0,
    lat_max: float = -40.0,
    lon_min: float = -70.0,
    lon_max: float = -40.0,
    output_dir: str = "dados_goes_roi",
    collection: str = "GOES19-L2-CMI-1",
    png: bool = False,
    items: list | None = None,
    lon_origin_graus: float = -75.2,
    intervalo_minutos: float | None = None,
) -> list[str]:
    """Lê os arquivos baixados do GOES, recorta a área de interesse,
    salva os arquivos resultantes no diretório de saída e apaga os originais.

    Itens cujo recorte final já exista em `output_dir` são pulados automaticamente.

    A ROI efetivamente usada (lat/lon min/max, longitude de origem e a coleção
    STAC) é gravada como atributos globais do arquivo recortado, para que
    `regradear_para_latlon` possa reaproveitá-la automaticamente depois, sem
    exigir que o chamador repita os mesmos valores manualmente numa segunda
    chamada.

    Args:
        banda (str): Identificador do asset/banda desejada (ex: 'B01' a 'B16').
        datetime_str (str): Data/hora única em ISO 8601 (AAAA-MM-DDTHH:MM:SSZ) ou intervalo temporal.
            Exemplos: '2026-06-01T14:30:00Z' ou '2026-06-01T14:00:00Z/2026-06-01T15:00:00Z'.
        lat_min (float, optional): Latitude mínima em graus decimais. Padrão é -20.0.
        lat_max (float, optional): Latitude máxima em graus decimais. Padrão é -40.0.
        lon_min (float, optional): Longitude mínima em graus decimais. Padrão é -70.0.
        lon_max (float, optional): Longitude máxima em graus decimais. Padrão é -40.0.
        output_dir (str, optional): Diretório local onde os arquivos recortados (.nc) 
            serão armazenados. Criado automaticamente caso não exista. Padrão é 'dados_goes_roi'.
        collection (str, optional): Coleção STAC a consultar quando `items` não é
            informado. Padrão é 'GOES19-L2-CMI-1'.
        png (bool, optional): Se True, gera imagens PNG dos recortes. Padrão é False.
        items (list, optional): Itens STAC já resolvidos (de `buscar_itens_goes`). Se
            None, a função faz a própria busca. Passar itens prontos evita repetir a
            consulta à API quando várias bandas do mesmo intervalo são processadas
            (ver `baixar_roi_multibanda_goes`).
        lon_origin_graus (float, optional): Longitude do ponto sub-satélite, repassada
            para `converter_latlon_para_xy`. Padrão é -75.2 (GOES-East).
        intervalo_minutos (float, optional): Intervalo entre as imagens, em minutos
            (ex: 60 para uma por hora). Usado apenas quando `items` não é informado.
            Padrão é None (todas).

    Returns:
        list[str]: Lista contendo os caminhos locais dos arquivos NetCDF recortados.

    Raises:
        FileNotFoundError: Se nenhum arquivo for baixado para realizar o recorte.
        Exception: Em caso de falha na leitura ou salvamento via xarray.

    Example:
        >>> arquivos_recortados = baixar_roi_goes(
        ...     banda="B14",
        ...     datetime_str="2026-06-01T14:30:00Z",
        ...     png=True
        ... )
        >>> print(arquivos_recortados)
        ['dados_goes_roi/crop_GOES19_CMI_..._B14.nc']
    """

    temp_dir = os.path.join(output_dir, "temp_downloads")
    os.makedirs(output_dir, exist_ok=True)
    arquivos_recortados = []

    # Resolve os itens da STAC
    if items is None:
        items = buscar_itens_goes(datetime_str, collection=collection, intervalo_minutos=intervalo_minutos)

    # Pula itens cujo recorte final já existe em disco: evita baixar de novo
    # um arquivo bruto que já foi processado em uma execução anterior.
    itens_pendentes = []
    for item in items:
        if banda not in item.assets:
            continue

        nome_bruto = f"{item.id}_{banda}.nc"
        path_destino = os.path.join(output_dir, f"crop_{nome_bruto}")

        if os.path.exists(path_destino):
            logger.info(f"Recorte já existe, pulando download: {path_destino}")
            arquivos_recortados.append(path_destino)

        else:
            itens_pendentes.append(item)

    if not itens_pendentes:
        logger.info("Todos os recortes já existem em disco. Nada para baixar.")
        
        return arquivos_recortados

    # Faz o download apenas dos arquivos brutos que ainda faltam
    arquivos_baixados = baixar_banda_goes(banda, datetime_str, output_dir=temp_dir, collection = collection, items=itens_pendentes)
    
    # Normaliza para caminhos absolutos e confirma existência em disco
    arquivos_baixados = [os.path.abspath(p) for p in arquivos_baixados]
    logger.info(f"Arquivos reportados pelo download: {arquivos_baixados}")

    arquivos_baixados = [p for p in arquivos_baixados if os.path.exists(p)]

    if not arquivos_baixados:
        logger.warning("Nenhum arquivo baixado para recortar (nenhum arquivo encontrado no disco).")
        
        return arquivos_recortados

    # Obtém os limites em radianos da Fixed Grid (x, y)
    x_min, x_max, y_min, y_max = converter_latlon_para_xy(
        lat_min, lat_max, lon_min, lon_max, lon_origin_graus=lon_origin_graus
    )

    # Processa, recorta e apaga cada arquivo original
    for path_original in arquivos_baixados:
        if not os.path.exists(path_original):
            continue

        nome_arquivo = os.path.basename(path_original)
        path_destino = os.path.join(output_dir, f"crop_{nome_arquivo}")

        logger.info(f"Recortando o arquivo: {nome_arquivo}")
        
        # Exceptions

        # Abre e carrega na memória para liberar o arquivo original.
        # Por padrão abre o arquivo original diretamente, só cai
        # para um caminho temporário ASCII se a abertura falhar por causa de
        # caracteres especiais no caminho.
        tmp_path = path_original
        try:
            try:
                with _NETCDF_LOCK, xr.open_dataset(path_original) as ds:
                    if ds.y[0] > ds.y[-1]:
                        ds_cropped = ds.sel(x=slice(x_min, x_max), y=slice(y_max, y_min)).load()

                    else:
                        ds_cropped = ds.sel(x=slice(x_min, x_max), y=slice(y_min, y_max)).load()

            except (UnicodeDecodeError, OSError) as e_open:
                logger.warning(f"Falha ao abrir {path_original} diretamente ({e_open}). Copiando para caminho temporário ASCII.")
                tmp_name = f"goes_tmp_{uuid.uuid4().hex}_{nome_arquivo}"
                tmp_path = os.path.join(tempfile.gettempdir(), tmp_name)
                shutil.copy2(path_original, tmp_path)

                with _NETCDF_LOCK, xr.open_dataset(tmp_path) as ds:
                    if ds.y[0] > ds.y[-1]:
                        ds_cropped = ds.sel(x=slice(x_min, x_max), y=slice(y_max, y_min)).load()

                    else:
                        ds_cropped = ds.sel(x=slice(x_min, x_max), y=slice(y_min, y_max)).load()

        except FileNotFoundError:
            logger.warning(f"Arquivo não encontrado ao abrir: {path_original}. Pulando.")

            # tenta remover o arquivo temporário se foi criado
            try:
                if tmp_path != path_original and os.path.exists(tmp_path):
                    os.remove(tmp_path)

            except Exception:
                pass

            continue

        except Exception as e:
            logger.error(f"Erro ao processar {path_original}: {e}. Pulando.")
            try:
                if tmp_path != path_original and os.path.exists(tmp_path):
                    os.remove(tmp_path)

            except Exception:
                pass

            continue

        finally:
            # remove o temporário se foi criado e diferente do original
            try:
                if 'tmp_path' in locals() and tmp_path != path_original and os.path.exists(tmp_path):
                    os.remove(tmp_path)

            except Exception:
                pass

        # Grava a ROI (e a coleção/série) usada como atributos globais do
        # recorte: permite que `regradear_para_latlon` reaproveite a mesma
        # ROI automaticamente, sem precisar repeti-la manualmente numa
        # segunda chamada (e sem risco de as duas ficarem dessincronizadas).
        ds_cropped.attrs["roi_lat_min"] = lat_min
        ds_cropped.attrs["roi_lat_max"] = lat_max
        ds_cropped.attrs["roi_lon_min"] = lon_min
        ds_cropped.attrs["roi_lon_max"] = lon_max
        ds_cropped.attrs["roi_lon_origin_graus"] = lon_origin_graus
        ds_cropped.attrs["colecao_stac"] = collection

        # Salva o arquivo recortado em um temporário ASCII e depois move para o destino final
        try:
            tmp_dest_name = f"crop_tmp_{uuid.uuid4().hex}_{nome_arquivo}"
            tmp_dest = os.path.join(tempfile.gettempdir(), tmp_dest_name)

            # escreve para o temporário (evita problemas com caminhos Unicode no Windows)
            with _NETCDF_LOCK:
                ds_cropped.to_netcdf(tmp_dest)

            # substitui destino final de forma atômica
            try:
                if os.path.exists(path_destino):
                    os.remove(path_destino)

            except Exception:
                pass
            
            os.replace(tmp_dest, path_destino)

            if png:
                try:
                    png_path = gerar_png_do_dataset(ds_cropped, path_destino)
                    logger.info(f"Imagem PNG gerada: {png_path}")

                except Exception as e:
                    logger.warning(f"Falha ao gerar PNG para {path_destino}: {e}")

            ds_cropped.close()
            logger.info(f"Arquivo recortado salvo em: {path_destino}")
            arquivos_recortados.append(path_destino)

        except PermissionError as e:
            logger.error(f"PermissionError ao salvar {path_destino}: {e}. Pulando este arquivo.")

            try:
                if 'tmp_dest' in locals() and os.path.exists(tmp_dest):
                    os.remove(tmp_dest)

            except Exception:
                pass

            continue
        except Exception as e:
            logger.error(f"Erro ao salvar recorte para {path_destino}: {e}. Pulando este arquivo.")

            try:
                if 'tmp_dest' in locals() and os.path.exists(tmp_dest):
                    os.remove(tmp_dest)

            except Exception:
                pass

            continue

        # Apaga o arquivo bruto baixado (apenas após salvar com sucesso)
        try:
            os.remove(path_original)
            logger.info(f"Arquivo original removido: {path_original}")

        except Exception as e:
            logger.warning(f"Não foi possível remover o arquivo original {path_original}: {e}")

    # Remove o diretório temporário se estiver vazio
    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
        os.rmdir(temp_dir)

    return arquivos_recortados

def _baixar_roi_multibanda_bloco_unico(
    datetime_str: str,
    bandas_lista: list[str],
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    output_dir: str,
    collection: str,
    png: bool,
    max_workers: int,
    lon_origin_graus: float,
    intervalo_minutos: float | None = None,
) -> list[str]:
    """Núcleo de `baixar_roi_multibanda_goes` para UM único intervalo/bloco de
    tempo: busca STAC (1x) + download + recorte de todas as bandas, em
    paralelo. Extraído como função à parte para ser chamado bloco a bloco por
    `baixar_roi_multibanda_goes` quando `tamanho_bloco_dias` é informado.
    """
    # Busca os itens STAC uma única vez
    items = buscar_itens_goes(datetime_str, collection=collection, intervalo_minutos=intervalo_minutos)
    if not items:
        return []

    def _processar_banda(band: str) -> list[str]:
        # Define um diretório específico para cada banda (ex: dados_goes_roi/B01)
        pasta_banda = os.path.join(output_dir, band)
        return baixar_roi_goes(
            banda=band,
            datetime_str=datetime_str,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            output_dir=pasta_banda,
            collection=collection,
            png=png,
            items=items,
            lon_origin_graus=lon_origin_graus,
        )

    todos_arquivos = []

    # Paraleliza as operações de I/O
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futuros = {executor.submit(_processar_banda, band): band for band in bandas_lista}

        for futuro in concurrent.futures.as_completed(futuros):
            band = futuros[futuro]

            try:
                todos_arquivos.extend(futuro.result())

            except Exception as e:
                logger.error(f"Erro ao processar a banda {band}: {e}")

    return todos_arquivos

def baixar_roi_multibanda_goes(
    datetime_str: str,
    bandas: list[str] = ["B01","B02","B03","B04","B05","B06","B07","B08","B09","B10","B11","B12","B13","B14","B15","B16"],
    lat_min: float = -20.0,
    lat_max: float = -40.0,
    lon_min: float = -70.0,
    lon_max: float = -40.0,
    output_dir: str = "dados_goes_roi",
    collection: str = "GOES19-L2-CMI-1",
    png: bool = False,
    max_workers: int = 4,
    lon_origin_graus: float = -75.2,
    tamanho_bloco_dias: int | None = None,
    intervalo_minutos: float | None = None,
) -> list[str]:
    """Baixa e recorta múltiplas bandas espectrais do GOES para a mesma janela
    de tempo e região de interesse (ROI), organizando a saída em uma subpasta por banda.

    Itens cujo recorte final já exista em disco são pulados automaticamente
    (nem chegam a ser baixados novamente), o que torna reexecuções do pipeline
    incrementais em vez de rebaixar tudo do zero. A busca no catálogo STAC é
    feita uma única vez por bloco, NA COLEÇÃO INFORMADA, e reaproveitada entre
    todas as bandas processadas.

    Args:
        datetime_str (str): Data/hora única em ISO 8601 (AAAA-MM-DDTHH:MM:SSZ) ou intervalo temporal.
            Exemplos: '2026-06-01T14:30:00Z' ou '2026-06-01T14:00:00Z/2026-06-01T15:00:00Z'.
        bandas (list[str], optional): Lista de identificadores de banda a processar
            (ex: 'B01' a 'B16' para o sensor ABI do GOES). Padrão é todas as 16 bandas.
        lat_min (float, optional): Latitude mínima em graus decimais. Padrão é -20.0.
        lat_max (float, optional): Latitude máxima em graus decimais. Padrão é -40.0.
        lon_min (float, optional): Longitude mínima em graus decimais. Padrão é -70.0.
        lon_max (float, optional): Longitude máxima em graus decimais. Padrão é -40.0.
        output_dir (str, optional): Diretório local onde os arquivos recortados (.nc)
            serão armazenados, em uma subpasta por banda (ex: 'dados_goes_roi/B01').
            Criado automaticamente caso não exista. Padrão é 'dados_goes_roi'.
        collection (str, optional): Coleção STAC a consultar - depende da série do
            satélite (ver `montar_colecao_stac`). Padrão é 'GOES19-L2-CMI-1'.
            Cobre uma única série/coleção por chamada: para históricos que
            atravessam uma troca de satélite, chame esta função uma vez para
            cada série/coleção.
        png (bool, optional): Se True, gera imagens PNG dos recortes. Padrão é False.
        max_workers (int, optional): Número de bandas processadas em paralelo
            (download + recorte são operações I/O-bound). Padrão é 4. Aumentar
            demais pode sobrecarregar a API do INPE; ajuste conforme a
            estabilidade da rede.
        lon_origin_graus (float, optional): Longitude do ponto sub-satélite, repassada
            para `converter_latlon_para_xy`. Padrão é -75.2 (GOES-East).
        tamanho_bloco_dias (int, optional): Se informado e `datetime_str` for um
            intervalo, divide-o internamente em blocos consecutivos de até
            essa quantidade de dias, processando um bloco por vez (ver
            `_dividir_intervalo_em_blocos`). Protege `buscar_itens_goes` de
            materializar uma lista enorme de itens de uma vez só (ex: anos de
            dados a ~10 min de cadência somam facilmente centenas de milhares
            de itens) e torna o processo retomável por bloco: um bloco com
            falha é registrado e não interrompe os demais. Padrão é None
            (sem divisão - comportamento igual ao anterior). Recomendado para
            períodos de mais de alguns dias/semanas (ex: 30 para backfills de
            meses/anos).
        intervalo_minutos (float, optional): Intervalo entre as imagens baixadas, em
            minutos. Padrão é None: todas as imagens do período (uma a cada ~10 min).
            Ex: 60 baixa uma imagem por hora (1/6 do volume), 30 uma a cada meia
            hora. Os horários-alvo são múltiplos do intervalo contados a partir de
            00:00 UTC (ver `_filtrar_por_intervalo`), e todas as bandas usam os
            mesmos instantes. Recomenda-se um valor que divida 1440 (10, 20, 30, 60,
            120, 180, 360, 720, 1440).

    Returns:
        list[str]: Lista contendo os caminhos locais de todos os arquivos NetCDF
            recortados, de todas as bandas (e blocos) processados.

    Raises:
        Exception: Em caso de falha de conexão, erro no cliente STAC ou erro na
            leitura/salvamento via xarray de alguma das bandas (quando
            `tamanho_bloco_dias` é None; com blocos, falhas por bloco são
            capturadas e apenas registradas em log).

    Example:
        >>> arquivos = baixar_roi_multibanda_goes(
        ...     datetime_str="2026-06-01T14:30:00Z",
        ...     bandas=["B13", "B14"],
        ...     png=True
        ... )
        >>> print(arquivos)
        ['dados_goes_roi/B13/crop_GOES19_CMI_..._B13.nc', 'dados_goes_roi/B14/crop_GOES19_CMI_..._B14.nc']

        >>> # Uma imagem por hora, em vez de uma a cada 10 min:
        >>> arquivos = baixar_roi_multibanda_goes(
        ...     datetime_str="2026-06-01T00:00:00Z/2026-06-08T00:00:00Z",
        ...     bandas=["B13", "B14"],
        ...     intervalo_minutos=60,
        ... )

        >>> # Backfill de vários anos, em blocos mensais:
        >>> arquivos = baixar_roi_multibanda_goes(
        ...     datetime_str="2020-01-01T00:00:00Z/2025-04-02T00:00:00Z",
        ...     collection="GOES16-L2-CMI-1",
        ...     tamanho_bloco_dias=30,
        ... )
    """

    if isinstance(bandas, str):
        bandas_lista = [bandas]

    else:
        bandas_lista = list(bandas or [])

    if not bandas_lista:
        logger.info("Nenhuma banda informada para processamento.")

        return []

    if tamanho_bloco_dias is not None:
        blocos = _dividir_intervalo_em_blocos(datetime_str, tamanho_bloco_dias)
    else:
        blocos = [datetime_str]

    todos_arquivos = []

    for bloco in blocos:
        if len(blocos) > 1:
            logger.info(f"=== Bloco de datas: {bloco} ===")

        try:
            todos_arquivos.extend(
                _baixar_roi_multibanda_bloco_unico(
                    datetime_str=bloco,
                    bandas_lista=bandas_lista,
                    lat_min=lat_min,
                    lat_max=lat_max,
                    lon_min=lon_min,
                    lon_max=lon_max,
                    output_dir=output_dir,
                    collection=collection,
                    png=png,
                    max_workers=max_workers,
                    lon_origin_graus=lon_origin_graus,
                    intervalo_minutos=intervalo_minutos,
                )
            )
        except Exception as e:
            # Um bloco com falha não deve derrubar um backfill inteiro de
            # vários anos - registra e segue para o próximo bloco.
            logger.error(f"Falha ao processar o bloco de datas {bloco}: {e}")

    return todos_arquivos
