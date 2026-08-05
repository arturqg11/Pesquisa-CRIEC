import os
import requests
from pystac_client import Client

def baixar_imagem_goes(banda, datetime_str, bbox, output_dir="dados_goes"):
    """
    Busca e baixa imagens do GOES-19 na STAC API do INPE.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    api_url = "https://data.inpe.br/bdc/stac/v1/"
    print(f"Conectando à STAC API: {api_url}")
    client = Client.open(api_url)
    
    print(f"Buscando dados para a coleção 'GOES19-L2-CMI-1'...")
    # Removido o parâmetro sortby para evitar o erro 400 da API do INPE
    search = client.search(
        collections=["GOES19-L2-CMI-1"],
        bbox=bbox,
        datetime=datetime_str
    )
    
    items = list(search.items())
    print(f"Total de registros encontrados: {len(items)}")
    
    if len(items) == 0:
        print("Nenhuma imagem encontrada para os parâmetros informados.")
        return

    # Usar um conjunto (set) para controlar IDs já baixados e evitar repetições
    ids_baixados = set()

    for item in items:
        if item.id in ids_baixados:
            continue
            
        print(f"\nProcessando Item ID: {item.id} (Data: {item.datetime})")
        
        if banda in item.assets:
            asset = item.assets[banda]
            file_url = asset.href
            
            file_name = f"{item.id}_{banda}.nc"
            file_path = os.path.join(output_dir, file_name)
            
            if os.path.exists(file_path):
                print(f"Arquivo já existe localmente: {file_name}. Pulando download.")
                ids_baixados.add(item.id)
                continue
                
            print(f"Baixando de: {file_url}")
            print(f"Salvando em: {file_path}...")
            
            response = requests.get(file_url, stream=True)
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"Download concluído com sucesso: {file_name}")
                ids_baixados.add(item.id)
            else:
                print(f"Erro ao baixar o arquivo. Status HTTP: {response.status_code}")
        else:
            print(f"Aviso: A banda '{banda}' não foi encontrada neste item.")

# ==========================================
# EXEMPLO DE USO
# ==========================================
if __name__ == "__main__":
    minha_regiao = [-48.0, -24.0, -43.0, -20.0]
    minha_data = "2026-06-01T14:30:00Z/2026-06-01T14:35:00Z"
    minha_banda = "B14" 
    
    baixar_imagem_goes(
        banda=minha_banda,
        datetime_str=minha_data,
        bbox=minha_regiao,
        output_dir="./imagens_goes_baixadas"
    )