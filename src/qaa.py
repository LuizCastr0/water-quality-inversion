# /src/qaa.py
# Implementação simplificada do QAA-RGB para Sentinel-2
# O objetivo é extrair IOPs (coeficientes de absorção e retroespalhamento) a partir das bandas RGB (B2, B3, B4) para melhorar a predição de clorofila-a.
#
import numpy as np

def qaa_rgb(rgb_reflectance):
    """
    QAA-RGB simplificado para Sentinel-2.
    
    Parâmetros:
        rgb_reflectance: array com 3 valores [Rrs_blue, Rrs_green, Rrs_red]
                         (bandas 2, 3, 4 do Sentinel-2)
    
    Retorna:
        a_total: coeficiente de absorção total em m^-1
        bb_total: coeficiente de retroespalhamento total em m^-1
    """
    # Constantes físicas (água pura, valores típicos para Sentinel-2)
    a_w = {
        'blue': 0.0045,   # 490nm (B2)
        'green': 0.0244,  # 560nm (B3)
        'red': 0.382      # 665nm (B4)
    }
    bb_w = 0.0024  # retroespalhamento da água pura (constante nas bandas visíveis)
    
    Rrs_blue, Rrs_green, Rrs_red = rgb_reflectance
    
    # Evitar divisão por zero
    eps = 1e-6
    
    # Passo 1: Calcular rrs (reflectância abaixo da superfície)
    rrs_blue = Rrs_blue / (0.52 + 1.7 * Rrs_blue)
    rrs_green = Rrs_green / (0.52 + 1.7 * Rrs_green)
    rrs_red = Rrs_red / (0.52 + 1.7 * Rrs_red)
    
    # Passo 2: Calcular u (razão bb/(a+bb)) para cada banda
    u_blue = (-0.0895 + np.sqrt(0.0895**2 + 4 * 0.1247 * rrs_blue)) / (2 * 0.1247)
    u_green = (-0.0895 + np.sqrt(0.0895**2 + 4 * 0.1247 * rrs_green)) / (2 * 0.1247)
    u_red = (-0.0895 + np.sqrt(0.0895**2 + 4 * 0.1247 * rrs_red)) / (2 * 0.1247)
    
    # Passo 3: Escolher banda de referência (assumir absorção dominada por água)
    # Usamos o vermelho como referência (assumindo que é a banda mais absorvida)
    a_red = a_w['red'] + 0.5  # estimativa inicial
    
    # Passo 4: Calcular bb_red
    bb_red = (u_red * a_red) / (1 - u_red)
    
    # Passo 5: Estimar slope espectral de bb
    gamma = np.log(bb_red / 0.0024) / np.log(665 / 560)
    
    # Passo 6: Calcular bb para outras bandas
    bb_blue = bb_red * (490 / 665) ** gamma
    bb_green = bb_red * (560 / 665) ** gamma
    
    # Passo 7: Calcular absorção para cada banda
    a_blue = (1 - u_blue) * bb_blue / u_blue
    a_green = (1 - u_green) * bb_green / u_green
    
    # Retornar as IOPs mais informativas para o modelo
    return {
        'a_blue': a_blue, 'a_green': a_green, 'a_red': a_red,
        'bb_blue': bb_blue, 'bb_green': bb_green, 'bb_red': bb_red,
        'ratio_bb_a_blue': bb_blue / (a_blue + eps),
        'ratio_bb_a_green': bb_green / (a_green + eps),
        'slope_gamma': gamma
    }


def compute_iops_from_patch(patch, band_indices):
    """
    Extrai as bandas RGB do patch e calcula as IOPs.
    
    Args:
        patch: array (n_bands, height, width)
        band_indices: dict com índices das bandas B2, B3, B4
    
    Returns:
        dicionário com IOPs calculadas
    """
    # Extrair valores centrais das bandas RGB
    center_h = patch.shape[1] // 2
    center_w = patch.shape[2] // 2
    
    rrs_blue = patch[band_indices['blue'], center_h, center_w]
    rrs_green = patch[band_indices['green'], center_h, center_w]
    rrs_red = patch[band_indices['red'], center_h, center_w]
    
    return qaa_rgb([rrs_blue, rrs_green, rrs_red])