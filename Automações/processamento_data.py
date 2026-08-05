import xarray as xr
import matplotlib.pyplot as plt
import os

def converter_nc_para_png(caminho_nc, caminho_png):
    """
    Lê um arquivo NetCDF (.nc) do GOES e o converte em uma imagem PNG.
    """
    print(f"Abrindo o arquivo NetCDF: {caminho_nc}")
    
    # Abre o dataset NetCDF
    with xr.open_dataset(caminho_nc) as ds:
        # O GOES geralmente armazena a variável principal de imagem com o nome 'CMI' 
        # (Cloud and Moisture Imagery)
        if 'CMI' in ds:
            dados = ds['CMI']
        else:
            # Caso o nome da variável principal seja diferente, pega a primeira variável 2D/3D disponível
            variavel_principal = list(ds.data_vars)[0]
            dados = ds[variavel_principal]
            print(f"Usando a variável: {variavel_principal}")

        print("Gerando a imagem PNG...")
        
        # Configurar o tamanho e a proporção da figura
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        
        # Plota os dados com uma escala de cinza (ótimo para infravermelho/banda 14)
        # Se for banda visível, 'gray' ou 'gist_earth' funcionam muito bem.
        im = ax.imshow(dados, cmap='gray', origin='upper')
        
        # Remover eixos e margens para salvar apenas a imagem limpa
        ax.axis('off')
        
        # Salvar em PNG com alta qualidade
        plt.savefig(caminho_png, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close(fig)
        
        print(f"Imagem PNG salva com sucesso em: {caminho_png}")

# ==========================================
# EXEMPLO DE USO
# ==========================================
if __name__ == "__main__":
    # Caminho do arquivo .nc que você baixou
    arquivo_nc = "./imagens_goes_baixadas/GOES19_L2_ABI_202606011430_B14.nc"
    
    # Caminho onde o PNG será salvo
    arquivo_png = "./imagens_goes_baixadas/GOES19_L2_ABI_202606011430_B1414.png"
    
    if os.path.exists(arquivo_nc):
        converter_nc_para_png(arquivo_nc, arquivo_png)
    else:
        print(f"Arquivo NetCDF não encontrado no caminho: {arquivo_nc}")