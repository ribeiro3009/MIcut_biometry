# dfiqi.py
# Implementação "DFIQI-like" baseada em:
# Swofford et al., "A method for measuring the quality of friction skin impression evidence:
# Method development and validation", Forensic Science International, 2021 (Open Access).
# Esta implementação reexecuta as fórmulas e parâmetros publicados (não é o executável original).

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

import numpy as np
import cv2


# -----------------------------
# Utilidades gerais
# -----------------------------

def _ensure_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _crop_square_roi(img: np.ndarray, cx: float, cy: float, size_px: int) -> np.ndarray:
    """
    Recorta ROI quadrada (size_px x size_px) centrada em (cx, cy).
    Se pegar bordas, 'espelha' a imagem para completar (BORDER_REFLECT_101).
    """
    h, w = img.shape[:2]
    half = size_px // 2
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    x1, y1 = x0 + size_px, y0 + size_px

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        img = cv2.copyMakeBorder(
            img, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REFLECT_101
        )
        x0 += pad_left
        y0 += pad_top
        x1 = x0 + size_px
        y1 = y0 + size_px

    return img[y0:y1, x0:x1]


# -----------------------------
# Segmentação local (ROI)
# -----------------------------

@dataclass
class SegmentationParams:
    adaptive_block: int = 15      # deve ser ímpar
    adaptive_C: int = 2           # deslocamento
    invert_before: bool = True    # paper usa imagem invertida 8-bit para processar



def segment_roi(roi_gray: np.ndarray, p: SegmentationParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segmenta ROI em 'signal' (cristas) e 'background' via limiar adaptativo.
    Retorna (roi_base_intensities, mask_signal_bool)
    - roi_base_intensities é a cópia (invertida ou não) na qual BS será calculado.
    """
    base = roi_gray
    if p.invert_before:
        base = 255 - base  # paper: "creates an inverted 8-bit copy" antes do processamento

    # ADAPTIVE_THRESH_GAUSSIAN_C com THRESH_BINARY (ridges ficam 255 após inversão)
    bw = cv2.adaptiveThreshold(
        base, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        max(p.adaptive_block | 1, 3),  # garante ímpar >= 3
        p.adaptive_C
    )
    mask_signal = bw > 0
    return base, mask_signal


# -----------------------------
# Variáveis locais (5)
# -----------------------------

def var_s3pg(mask_signal: np.ndarray) -> float:
    """S3PG = % de pixels 'signal' na ROI (0..100)."""
    total = mask_signal.size
    if total == 0:
        return 0.0
    return 100.0 * (mask_signal.sum() / float(total))


def var_bimodal_separation(base: np.ndarray, mask_signal: np.ndarray) -> float:
    """
    BS = (mu_S - mu_B) / (2*(sigma_S + sigma_B))  [Eq.1 no artigo]
    Calculado nas intensidades da cópia base (invertida se p.invert_before=True).
    """
    S = base[mask_signal]
    B = base[~mask_signal]
    if S.size == 0 or B.size == 0:
        return 0.0
    mu_S, mu_B = float(S.mean()), float(B.mean())
    sig_S = float(S.std(ddof=0))
    sig_B = float(B.std(ddof=0))
    denom = 2.0 * (sig_S + sig_B) + 1e-9
    return (mu_S - mu_B) / denom


def var_acutance(roi_gray: np.ndarray) -> float:
    """
    ACUT = ln( mean_{todas janelas 3x3} sum_{8 vizinhos} (Ic - In)^2 / 8 )   [Eq.2]
    Implementação vetorizada: soma os 8 deslocamentos e normaliza por número de centros (p-2)^2.
    """
    I = roi_gray.astype(np.float32)
    H, W = I.shape
    if H < 3 or W < 3:
        return 0.0

    # região de centros válidos
    C = I[1:-1, 1:-1]
    acc = np.zeros_like(C, dtype=np.float32)

    # 8 deslocamentos vizinhos
    neighbors = [
        I[0:-2, 0:-2], I[0:-2, 1:-1], I[0:-2, 2:  ],
        I[1:-1, 0:-2],                I[1:-1, 2:  ],
        I[2:  , 0:-2], I[2:  , 1:-1], I[2:  , 2:  ],
    ]
    for N in neighbors:
        acc += (C - N) ** 2

    # média por janela (divide por 8) e média sobre todos centros
    acc_mean = acc.mean() / 8.0
    # log natural
    return float(np.log(max(acc_mean, 1e-9)))


def var_mean_object_width(mask_signal: np.ndarray) -> float:
    """
    MOW = média da largura (eixo menor) das elipses ajustadas aos objetos (componentes) de 'signal'.
    Retorna em 'pixels' da ROI.
    """
    m = (mask_signal.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    widths = []

    for cnt in contours:
        if len(cnt) < 5:
            continue
        (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)  # MA=major axis, ma=minor axis (comprimentos)
        minor = min(MA, ma)
        if minor > 0:
            widths.append(float(minor))

    if not widths:
        return 0.0
    return float(np.mean(widths))


def var_spatial_frequency(roi_gray: np.ndarray, dpi: float) -> float:
    """
    SF = frequência espacial das cristas (ridges/mm) via pico dominante na FFT 2D (fora do DC).
    """
    I = roi_gray.astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2(I))
    M = np.abs(F)

    H, W = I.shape
    cy, cx = H // 2, W // 2

    # suprime DC numa vizinhança central
    rr, cc = np.ogrid[:H, :W]
    R = np.sqrt((rr - cy) ** 2 + (cc - cx) ** 2)
    M[R <= 3.0] = 0.0

    # acha pico dominante
    iy, ix = np.unravel_index(np.argmax(M), M.shape)
    # frequência normalizada (ciclos/pixel)
    fy = abs(iy - cy) / float(H)
    fx = abs(ix - cx) / float(W)
    f_rad = math.hypot(fx, fy)  # magnitude radial

    # converte para ridges por milímetro
    px_per_mm = dpi / 25.4
    return float(f_rad * px_per_mm)


# -----------------------------
# Normalização (scores) e LQS
# -----------------------------

@dataclass(frozen=True)
class ScoringParams:
    # Tabela 1 do artigo (parâmetros de localização e escala)
    mu_S3PG: float = 51.408
    sigma_S3PG: float = 4.134

    mu_BS: float = 0.843
    sigma_BS: float = 0.147

    mu_ACUT: float = 6.869
    b_ACUT: float = 0.532  # escala logística (bs)

    mu_MOW: float = 1.383
    sigma_MOW: float = 0.391

    mu_SF: float = 2.078
    sigma_SF: float = 0.397



def score_gaussian(x: float, mu: float, sigma: float) -> float:
    """f(x) = exp(- (x - mu)^2 / (2*sigma^2))  ∈ (0,1]"""
    if sigma <= 0:
        return 0.0
    z2 = (x - mu) ** 2 / (2.0 * (sigma ** 2))
    return float(np.exp(-z2))


def score_logistic_cdf(x: float, mu: float, b: float) -> float:
    """g(x) = 1 / (1 + exp(-(x - mu)/b))  ∈ (0,1) — CDF logística (crescente em x)."""
    if b <= 0:
        return 0.0
    return float(1.0 / (1.0 + math.exp(-(x - mu) / b)))


def compute_lqs_for_roi(
    roi_gray: np.ndarray,
    dpi: float,
    seg_params: SegmentationParams,
    scoring: ScoringParams
) -> Dict[str, float]:
    """
    Calcula as 5 variáveis na ROI, normaliza (scores) e devolve LQS (média dos 5).
    """
    base, mask_signal = segment_roi(roi_gray, seg_params)

    s3pg = var_s3pg(mask_signal)
    bs   = var_bimodal_separation(base, mask_signal)
    acut = var_acutance(roi_gray)  # paper: acutance no ROI não segmentado
    mow  = var_mean_object_width(mask_signal)
    sf   = var_spatial_frequency(roi_gray, dpi)

    # normalização
    s_s3pg = score_gaussian(s3pg, scoring.mu_S3PG, scoring.sigma_S3PG)
    s_bs   = score_gaussian(bs,   scoring.mu_BS,   scoring.sigma_BS)
    s_acut = score_logistic_cdf(acut, scoring.mu_ACUT, scoring.b_ACUT)
    s_mow  = score_gaussian(mow,  scoring.mu_MOW,  scoring.sigma_MOW)
    s_sf   = score_gaussian(sf,   scoring.mu_SF,   scoring.sigma_SF)

    lqs = float(np.mean([s_s3pg, s_bs, s_acut, s_mow, s_sf]))

    return {
        "S3PG": s3pg, "BS": bs, "ACUT": acut, "MOW": mow, "SF": sf,
        "S_S3PG": s_s3pg, "S_BS": s_bs, "S_ACUT": s_acut, "S_MOW": s_mow, "S_SF": s_sf,
        "LQS": lqs
    }


# -----------------------------
# GQS (Value/Complexity/Difficulty)
# -----------------------------

@dataclass(frozen=True)
class MultinomialModel:
    """
    Modelo multinomial no formato "baseline com coeficientes zero".
    classes em ordem; para a classe baseline os coeficientes são (0,0,0).
    Cada outra classe tem (intercept, beta_LQSsum, beta_nFEAT).
    """
    classes: Tuple[str, ...]
    coefs: Dict[str, Tuple[float, float, float]]  # por classe


# Tabelas 2a, 2b, 2c do paper
VALUE_MODEL = MultinomialModel(
    classes=("NoValue", "ValueExclusion", "ValueIdentification"),
    coefs={
        "NoValue": (0.0, 0.0, 0.0),
        "ValueExclusion": (-1.736, -0.051, 0.277),
        "ValueIdentification": (-6.042, 0.495, 0.726),
    }
)

COMPLEXITY_MODEL = MultinomialModel(
    classes=("HighlyComplex", "Complex", "NonComplex"),
    coefs={
        "Complex": (0.0, 0.0, 0.0),
        "HighlyComplex": (3.325, -0.100, -0.459),
        "NonComplex": (-1.781, 0.741, -0.025),
    }
)

DIFFICULTY_MODEL = MultinomialModel(
    classes=("High", "Medium", "Low"),
    coefs={
        "High": (0.0, 0.0, 0.0),
        "Medium": (-1.896, 0.289, 0.125),
        "Low": (-3.071, 0.965, -0.004),
    }
)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def multinomial_probs(model: MultinomialModel, LQSsum: float, nFEAT: int) -> Dict[str, float]:
    """
    Constrói logits relativos à classe baseline (cujo vetor é (0,0,0)) e aplica softmax.
    """
    logits = []
    for cls in model.classes:
        b0, bL, bN = model.coefs[cls]
        logits.append(b0 + bL * LQSsum + bN * nFEAT)
    logits = np.array(logits, dtype=np.float64)
    probs = _softmax(logits)
    return {cls: float(p) for cls, p in zip(model.classes, probs)}


def compute_gqs(lqs_list: List[float]) -> Dict[str, Any]:
    """
    Aglutina LQS por impressão:
      - LQSsum = soma dos LQS individuais
      - nFEAT  = quantidade de features
      - Probabilidades multinomiais por determinação
      - GQS conforme Eq. 6–8 (diferença de extremos)
    """
    n = len(lqs_list)
    LQSsum = float(np.sum(lqs_list)) if n else 0.0
    nFEAT = int(n)

    # Probabilidades
    pv = multinomial_probs(VALUE_MODEL, LQSsum, nFEAT)
    pc = multinomial_probs(COMPLEXITY_MODEL, LQSsum, nFEAT)
    pd = multinomial_probs(DIFFICULTY_MODEL, LQSsum, nFEAT)

    # GQS (diferença de extremos) – Eq. 6–8
    value_gqs = pv["ValueIdentification"] - pv["NoValue"]
    compl_gqs = pc["NonComplex"] - pc["HighlyComplex"]
    diffi_gqs = pd["Low"] - pd["High"]

    return {
        "nFEAT": nFEAT,
        "LQSsum": LQSsum,
        "probs_value": pv,
        "probs_complexity": pc,
        "probs_difficulty": pd,
        "ValueGQS": float(value_gqs),
        "ComplexityGQS": float(compl_gqs),
        "DifficultyGQS": float(diffi_gqs),
    }


# -----------------------------
# Função principal por impressão
# -----------------------------

@dataclass
class DFIQIParams:
    dpi: float = 500.0
    roi_inch: float = 0.1
    # antes era: seg = SegmentationParams() / scoring = ScoringParams()
    # ⬇️ O CERTO é usar default_factory (NÃO use SegmentationParams ou SegmentationParams())
    seg: SegmentationParams = field(default_factory=SegmentationParams)
    # ⬇️ Mesmo que ScoringParams seja frozen, mantenha default_factory
    scoring: ScoringParams = field(default_factory=ScoringParams)



def dfiqi_on_image(
    img: np.ndarray,
    features_xy: List[Tuple[float, float]],
    params: DFIQIParams = DFIQIParams()
) -> Dict[str, Any]:
    """
    Aplica o pipeline DFIQI-like numa imagem:
      - Para cada feature (x,y), recorta uma ROI de 0,1" e calcula LQS
      - Agrega LQS para GQS (Value/Complexity/Difficulty)

    Retorna:
      {
        "per_feature": [ { "x":..., "y":..., ...variáveis..., "LQS":... }, ... ],
        "global": { "nFEAT", "LQSsum", "probs_*", "ValueGQS", "ComplexityGQS", "DifficultyGQS" }
      }
    """
    gray = _ensure_gray_u8(img)
    size_px = max(3, int(round(params.roi_inch * params.dpi)))
    if size_px % 2 == 0:
        size_px += 1  # centralizar exatamente

    per_feature = []
    for (x, y) in features_xy:
        roi = _crop_square_roi(gray, x, y, size_px)
        lqs_info = compute_lqs_for_roi(roi, params.dpi, params.seg, params.scoring)
        item = {"x": float(x), "y": float(y), **lqs_info}
        per_feature.append(item)

    g = compute_gqs([pf["LQS"] for pf in per_feature])

    return {"per_feature": per_feature, "global": g}
