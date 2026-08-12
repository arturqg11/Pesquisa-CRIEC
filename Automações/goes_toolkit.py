# Imports
import os
import requests
from pystac_client import Client
import shutil
import tempfile
import uuid
import concurrent.futures
import threading

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_NETCDF_LOCK = threading.Lock()

# Funções

def gerar_png_do_dataset(ds, path_nc: str) -> str:
    """Gera um PNG a partir de um dataset xarray já carregado em memória."""

    if "CMI" in ds.data_vars:
        var_name = "CMI"
    else:
        var_name = next(iter(ds.data_vars), None)

    if var_name is None:
        raise ValueError(f"Nenhuma variável de imagem encontrada em {path_nc}")

    plt.figure(figsize=(8, 8))
    plt.imshow(ds[var_name])
    png_path = path_nc.replace('.nc', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    return png_path

def buscar_itens_goes(
    datetime_str: str,
    collection: str = "GOES19-L2-CMI-1"
) -> list:
    """Consulta a STAC API do INPE uma única vez e retorna os itens encontrados.

    O resultado da busca depende apenas do intervalo de tempo (e da coleção),
    não da banda espectral. Centralizar a busca aqui permite reaproveitar os
    mesmos itens ao processar várias bandas (ver `roi_mbanda`), evitando
    repetir a mesma consulta à API STAC uma vez por banda.

    Args:
        datetime_str (str): Data/hora única em ISO 8601 (AAAA-MM-DDTHH:MM:SSZ) ou intervalo temporal.
        collection (str, optional): Coleção STAC a consultar. Padrão é 'GOES19-L2-CMI-1'.

    Returns:
        list: Itens (pystac.Item) encontrados no catálogo.
    """
    api_url = "https://data.inpe.br/bdc/stac/v1/"
    print(f"Conectando à STAC API: {api_url}")
    client = Client.open(api_url)

    print(f"Buscando dados para a coleção '{collection}'...")
    search = client.search(collections=[collection], datetime=datetime_str)

    items = list(search.items())
    print(f"Total de registros encontrados: {len(items)}")
    return items

def baixar_banda_goes(
    banda: str,
    datetime_str: str,
    output_dir: str = "dados_goes",
    items: list | None = None
) -> list[str]:
    """Busca e realiza o download dos dados em NetCDF do GOES-19 na STAC API do INPE.

    A função consulta o catálogo STAC do Brazil Data Cube (BDC/INPE) para a 
    coleção de dados de reflectância e temperatura do GOES-19 (`GOES19-L2-CMI-1`), 
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

    Returns:
        list[str]: Lista contendo os caminhos locais de todos os arquivos baixados 
            ou já existentes no disco.

    Raises:
        requests.exceptions.RequestException: Em caso de erro de conexão ao realizar o download.
        Exception: Em caso de falha de conexão ou erro no cliente STAC.

    Example:
        >>> baixados = baixar_imagem_goes(
        ...     banda="B14",
        ...     datetime_str="2026-06-01T14:30:00Z/2026-06-01T14:35:00Z",
        ...     output_dir="./dados_goes"
        ... )
        >>> print(baixados)
        ['./dados_goes/GOES19_CMI_..._B14.nc']
    """
    os.makedirs(output_dir, exist_ok=True)

    if items is None:
        items = buscar_itens_goes(datetime_str)

    if len(items) == 0:
        print("Nenhuma imagem encontrada para os parâmetros informados.")
        return []

    ids_baixados = set()
    arquivos_salvos = []

    for item in items:
        if item.id in ids_baixados:
            continue
            
        print(f"\nProcessando Item ID: {item.id} (Data: {item.datetime})")
        
        if banda in item.assets:
            asset = item.assets[banda]
            file_url = asset.href
            
            file_name = f"{item.id}_{banda}.nc"
            file_path = os.path.abspath(os.path.join(output_dir, file_name))
            
            if os.path.exists(file_path):
                print(f"Arquivo já existe localmente: {file_name}. Pulando download.")
                ids_baixados.add(item.id)
                arquivos_salvos.append(file_path)
                continue
                
            print(f"Baixando de: {file_url}")
            print(f"Salvando em: {file_path}...")
            
            response = requests.get(file_url, stream=True, timeout=(10, 300))
            if response.status_code == 200:
                # garante que o diretório existe antes de escrever
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                # confirma que o arquivo foi escrito no disco e usa caminho absoluto
                if os.path.exists(file_path):
                    print(f"Download concluído com sucesso: {file_name}")
                    ids_baixados.add(item.id)
                    arquivos_salvos.append(file_path)
                else:
                    print(f"Falha ao salvar o arquivo: {file_path}")
            else:
                print(f"Erro ao baixar o arquivo. Status HTTP: {response.status_code}")
        else:
            print(f"Aviso: A banda '{banda}' não foi encontrada neste item.")

    return arquivos_salvos

def conver_coord(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float
) -> tuple[float, float, float, float]:
    """Converte limites geográficos (Lat/Lon) nos limites de varredura (x, y) do GOES-19.

    A partir das coordenadas de uma caixa delimitadora em graus decimais 
    (lat_min, lat_max, lon_min, lon_max), a função utiliza as constantes orbitais 
    do GOES-19 para calcular os limites correspondentes na Fixed Grid (GEOS) 
    do satélite.

    Args:
        lat_min (float): Latitude mínima (Sul) em graus decimais.
        lat_max (float): Latitude máxima (Norte) em graus decimais.
        lon_min (float): Longitude mínima (Oeste) em graus decimais.
        lon_max (float): Longitude máxima (Leste) em graus decimais.

    Returns:
        tuple[float, float, float, float]: Tupla contendo os limites em radianos:
            (x_min, x_max, y_min, y_max)

    Example:
        >>> x_min, x_max, y_min, y_max = conver_coord(
        ...     lat_min=-24.0, lat_max=-20.0,
        ...     lon_min=-48.0, lon_max=-43.0
        ... )
        >>> print(f"x: [{x_min:.6f}, {x_max:.6f}], y: [{y_min:.6f}, {y_max:.6f}]")
    """

    # Constantes geodésicas e de projeção do GOES-19 (PUG Volume 5)
    R_EQ_GOES19 = 6378137.0          # Raio equatorial em metros (GRS80)
    R_POL_GOES19 = 6356752.31414      # Raio polar em metros
    H_SAT_GOES19 = 35786023.0        # Altura de perspectiva do satélite em metros
    H_DIST_GOES19 = H_SAT_GOES19 + R_EQ_GOES19  # Distância do centro da Terra ao satélite (m)
    LON_ORIGIN_GOES19 = -75.2        # Longitude do ponto sub-satélite em graus (GOES-East)

    # 1. Define os 4 cantos da caixa delimitadora em Lat/Lon
    lats = np.array([lat_min, lat_min, lat_max, lat_max])
    lons = np.array([lon_min, lon_max, lon_min, lon_max])

    # 2. Converte graus decimais para radianos
    phi = np.radians(lats)
    lamb = np.radians(lons)
    lambda_0 = np.radians(LON_ORIGIN_GOES19)

    # 3. Cálculos de geometria elipsoidal
    e2 = (R_EQ_GOES19**2 - R_POL_GOES19**2) / (R_EQ_GOES19**2)
    phi_c = np.arctan((R_POL_GOES19**2 / R_EQ_GOES19**2) * np.tan(phi))
    r_c = R_POL_GOES19 / np.sqrt(1 - e2 * np.cos(phi_c)**2)

    sx = H_DIST_GOES19 - r_c * np.cos(phi_c) * np.cos(lamb - lambda_0)
    sy = -r_c * np.cos(phi_c) * np.sin(lamb - lambda_0)
    sz = r_c * np.sin(phi_c)

    # 4. Ângulos de varredura em radianos (inversão do sinal em sy e uso de asin para y)
    x_angles = np.arctan(-sy / sx)
    y_angles = np.asin(sz / np.sqrt(sx**2 + sy**2 + sz**2))

    # 5. Extrai os limites mínimos e máximos da região
    x_min, x_max = float(np.min(x_angles)), float(np.max(x_angles))
    y_min, y_max = float(np.min(y_angles)), float(np.max(y_angles))

    return x_min, x_max, y_min, y_max

def baixar_roi_goes(
    banda: str,
    datetime_str: str,
    lat_min: float = -20.0,
    lat_max: float = -40.0,
    lon_min: float = -70.0,
    lon_max: float = -40.0,
    output_dir: str = "dados_goes_roi",
    png: bool = False,
    items: list | None = None
) -> list[str]:
    """Lê os arquivos baixados do GOES-19, recorta a área de interesse usando xarray,
    salva os arquivos resultantes no diretório de saída e apaga os originais.

    Itens cujo recorte final já exista em `output_dir` são pulados automaticamente
    (nem chegam a ser baixados novamente), o que torna reexecuções do pipeline
    incrementais em vez de rebaixar tudo do zero.

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
        png (bool, optional): Se True, gera imagens PNG dos recortes. Padrão é False.
        items (list, optional): Itens STAC já resolvidos (de `buscar_itens_goes`). Se
            None, a função faz a própria busca. Passar itens prontos evita repetir a
            consulta à API quando várias bandas do mesmo intervalo são processadas
            (ver `roi_mbanda`).

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

    # 0. Resolve os itens STAC (reaproveita se já vierem prontos de roi_mbanda)
    if items is None:
        items = buscar_itens_goes(datetime_str)

    # 0.1 Pula itens cujo recorte final já existe em disco: evita baixar de novo
    #     um arquivo bruto que já foi processado em uma execução anterior.
    itens_pendentes = []
    for item in items:
        if banda not in item.assets:
            continue
        nome_bruto = f"{item.id}_{banda}.nc"
        path_destino = os.path.join(output_dir, f"crop_{nome_bruto}")
        if os.path.exists(path_destino):
            print(f"Recorte já existe, pulando download: {path_destino}")
            arquivos_recortados.append(path_destino)
        else:
            itens_pendentes.append(item)

    if not itens_pendentes:
        print("Todos os recortes já existem em disco. Nada para baixar.")
        return arquivos_recortados

    # 1. Faz o download apenas dos arquivos brutos que ainda faltam
    arquivos_baixados = baixar_banda_goes(banda, datetime_str, output_dir=temp_dir, items=itens_pendentes)
    
    # Normaliza para caminhos absolutos e confirma existência em disco
    arquivos_baixados = [os.path.abspath(p) for p in arquivos_baixados]
    print(f"Arquivos reportados pelo download: {arquivos_baixados}")
    arquivos_baixados = [p for p in arquivos_baixados if os.path.exists(p)]
    if not arquivos_baixados:
        print("Nenhum arquivo baixado para recortar (nenhum arquivo encontrado no disco).")
        return arquivos_recortados

    # 2. Obtém os limites em radianos da Fixed Grid (x, y)
    x_min, x_max, y_min, y_max = conver_coord(lat_min, lat_max, lon_min, lon_max)

    # 3. Processa, recorta e apaga cada arquivo original
    for path_original in arquivos_baixados:
        if not os.path.exists(path_original):
            continue

        nome_arquivo = os.path.basename(path_original)
        path_destino = os.path.join(output_dir, f"crop_{nome_arquivo}")

        print(f"\nRecortando o arquivo: {nome_arquivo}")
        
        # Abre e carrega na memória para liberar o arquivo original imediatamente.
        # Por padrão abre o arquivo original diretamente (sem cópia extra); só cai
        # para um caminho temporário ASCII se a abertura falhar por causa de
        # caracteres especiais no caminho (problema conhecido no Windows).
        tmp_path = path_original
        try:
            try:
                with _NETCDF_LOCK, xr.open_dataset(path_original) as ds:
                    if ds.y[0] > ds.y[-1]:
                        ds_cropped = ds.sel(x=slice(x_min, x_max), y=slice(y_max, y_min)).load()
                    else:
                        ds_cropped = ds.sel(x=slice(x_min, x_max), y=slice(y_min, y_max)).load()
            except (UnicodeDecodeError, OSError) as e_open:
                print(f"Aviso: falha ao abrir {path_original} diretamente ({e_open}). Copiando para caminho temporário ASCII.")
                tmp_name = f"goes_tmp_{uuid.uuid4().hex}_{nome_arquivo}"
                tmp_path = os.path.join(tempfile.gettempdir(), tmp_name)
                shutil.copy2(path_original, tmp_path)
                with _NETCDF_LOCK, xr.open_dataset(tmp_path) as ds:
                    if ds.y[0] > ds.y[-1]:
                        ds_cropped = ds.sel(x=slice(x_min, x_max), y=slice(y_max, y_min)).load()
                    else:
                        ds_cropped = ds.sel(x=slice(x_min, x_max), y=slice(y_min, y_max)).load()
        except FileNotFoundError:
            print(f"Aviso: arquivo não encontrado ao abrir: {path_original}. Pulando.")
            # tenta remover o arquivo temporário se foi criado
            try:
                if tmp_path != path_original and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            continue
        except Exception as e:
            print(f"Erro ao processar {path_original}: {e}. Pulando.")
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
                    print(f"Imagem PNG gerada: {png_path}")
                except Exception as e:
                    print(f"Aviso: falha ao gerar PNG para {path_destino}: {e}")

            ds_cropped.close()
            print(f"Arquivo recortado salvo em: {path_destino}")
            arquivos_recortados.append(path_destino)
        except PermissionError as e:
            print(f"PermissionError ao salvar {path_destino}: {e}. Pulando este arquivo.")
            try:
                if 'tmp_dest' in locals() and os.path.exists(tmp_dest):
                    os.remove(tmp_dest)
            except Exception:
                pass
            continue
        except Exception as e:
            print(f"Erro ao salvar recorte para {path_destino}: {e}. Pulando este arquivo.")
            try:
                if 'tmp_dest' in locals() and os.path.exists(tmp_dest):
                    os.remove(tmp_dest)
            except Exception:
                pass
            continue

        # Apaga o arquivo bruto baixado (apenas após salvar com sucesso)
        try:
            os.remove(path_original)
            print(f"Arquivo original removido: {path_original}")
        except Exception as e:
            print(f"Aviso: não foi possível remover o arquivo original {path_original}: {e}")

    # Remove o diretório temporário se estiver vazio
    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
        os.rmdir(temp_dir)

    return arquivos_recortados

def roi_mbanda(
    datetime_str: str,
    bandas: list[str] = ["B01","B02","B03","B04","B05","B06","B07","B08","B09","B10","B11","B12","B13","B14","B15","B16"],
    lat_min: float = -20.0,
    lat_max: float = -40.0,
    lon_min: float = -70.0,
    lon_max: float = -40.0,
    output_dir: str = "dados_goes_roi",
    png: bool = False,
    max_workers: int = 4
) -> list[str]:
    """Baixa e recorta múltiplas bandas espectrais do GOES-19 para a mesma janela
    de tempo e região de interesse (ROI), organizando a saída em uma subpasta por banda.

    Itens cujo recorte final já exista em disco são pulados automaticamente
    (nem chegam a ser baixados novamente), o que torna reexecuções do pipeline
    incrementais em vez de rebaixar tudo do zero. A busca no catálogo STAC é
    feita uma única vez e reaproveitada entre todas as bandas processadas.

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
        png (bool, optional): Se True, gera imagens PNG dos recortes. Padrão é False.
        max_workers (int, optional): Número de bandas processadas em paralelo
            (download + recorte são operações I/O-bound). Padrão é 4. Aumentar
            demais pode sobrecarregar a API do INPE; ajuste conforme a
            estabilidade da rede.

    Returns:
        list[str]: Lista contendo os caminhos locais de todos os arquivos NetCDF
            recortados, de todas as bandas processadas.

    Raises:
        Exception: Em caso de falha de conexão, erro no cliente STAC ou erro na
            leitura/salvamento via xarray de alguma das bandas.

    Example:
        >>> arquivos = roi_mbanda(
        ...     datetime_str="2026-06-01T14:30:00Z",
        ...     bandas=["B13", "B14"],
        ...     png=True
        ... )
        >>> print(arquivos)
        ['dados_goes_roi/B13/crop_GOES19_CMI_..._B13.nc', 'dados_goes_roi/B14/crop_GOES19_CMI_..._B14.nc']
    """
    if isinstance(bandas, str):
        bandas_lista = [bandas]
    else:
        bandas_lista = list(bandas or [])

    if not bandas_lista:
        print("Nenhuma banda informada para processamento.")
        return []

    # Busca os itens STAC uma única vez e reaproveita para todas as bandas: o
    # resultado da busca não depende da banda, só do intervalo de tempo.
    items = buscar_itens_goes(datetime_str)
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
            png=png,
            items=items
        )

    todos_arquivos = []

    # Download/recorte são operações de I/O (rede + disco), então paralelizar
    # com threads reduz bastante o tempo total ao processar várias bandas.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futuros = {executor.submit(_processar_banda, band): band for band in bandas_lista}
        for futuro in concurrent.futures.as_completed(futuros):
            band = futuros[futuro]
            try:
                todos_arquivos.extend(futuro.result())
            except Exception as e:
                print(f"Erro ao processar a banda {band}: {e}")

    return todos_arquivos