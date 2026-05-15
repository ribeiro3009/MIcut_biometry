# INVENTARIO_FEATURES.md

Inventário das features de qualidade de impressão digital implementadas no `micut_biometry`, mapeadas para o contrato `IMIcutQualityExtractor` do coletor `ColetorQualidades.WinForms` (.NET 8 / x86).

---

## 1. Visão de alto nível

- **Linguagem do projeto atual:** Python 3 (numpy 2.1, opencv-contrib 4.12, torch 2.5+cu121, scikit-learn 1.7, jpype1 1.6, cbor2, polars).
- **Entrada do pipeline:** BMPs nomeados `<ID>_dedo<N>.bmp` (N = 1..10). O pipeline em `src/pipeline.py` faz **muito mais** do que calcular features por dedo: concatena dedos em "colunas", roda um Faster R-CNN para segmentar, recorta cada dedo em 500 DPI e gera máscaras ROI. **Para o coletor, todo esse pré-processamento é irrelevante** — o coletor já entrega BMPs por dedo (FC3 alimentou o BMP via Dermalog). A camada de segmentação ML serve apenas ao caso de uso original (digitais escaneadas em folha-decadatilar). Portanto, o "input efetivo" das features individuais é o **BMP 500 DPI de um único dedo + máscara ROI binária**.
- **Máscara ROI:** atualmente gerada pelo `remove_lines_keep_fingerprints` em `ml_segmentation.py`. Algumas features (shape, frequency, texture/orientation_std) consomem essa máscara. Sem a máscara, `texture.analyze_texture` tem fallback via Otsu (linhas 88–97 de `texture.py`), mas `shape.analyze_shape` **não tem fallback** e `frequency.analyze_ridge_frequency` aceita `roi_mask=None` (analisa a imagem inteira sem filtrar).
- **Linguagem-alvo:** .NET 8 x86. Como o motor é Python + Java (SourceAFIS via JPype) + binários NIST, **a maioria das features cai no padrão "subprocess CLI"** descrito no Apêndice A do prompt. Tentar embutir Python/Java na DLL é inviável dentro do escopo.

---

## 2. Features candidatas a porte

Total: **9 features extraídas por dedo no `micut_biometry`** (mais o NFIQ2, que o coletor já tem). Listadas em ordem de aparição no CSV (`docs/TECHNICAL.md` §4).

### 2.1 `minutiae_count` — quantidade de minúcias

- **Nome curto sugerido:** `MIcutMinutiaeCount`
- **O que mede:** número total de minúcias detectadas pelo SourceAFIS sobre o BMP do dedo (a 500 DPI). Score mais alto = digital mais rica em pontos característicos, geralmente correlato a área útil + nitidez.
- **Faixa de saída:** inteiro ≥ 0. Empiricamente, dedos limpos a 500 DPI ficam em 30–120; <10 é digital muito pobre/borrada. **Quantização não necessária** (já é `int`). Sugiro `ScoreMaximo = 200` para evitar saturação em outliers.
- **Entrada exigida:** grayscale 8-bit do dedo, com DPI = 500. **Não precisa de máscara ROI.**
- **Dependências runtime:**
  - JVM (Java 8+) com os JARs em `bin/`:
    - `sourceafis-3.18.1.jar` (181 KB)
    - `fingerprintio-1.3.3.jar` (160 KB)
    - `closeablescope-1.0.1.jar` (4.5 KB)
    - `commons-io-2.11.0.jar` (327 KB)
    - `jackson-annotations-2.15.2.jar` (75 KB)
    - `jackson-core-2.15.2.jar` (549 KB)
    - `jackson-databind-2.15.2.jar` (1.6 MB)
    - `jackson-dataformat-cbor-2.15.2.jar` (69 KB)
    - `noexception-1.9.1.jar` (252 KB)
    - `slf4j-api-1.7.32.jar` (41 KB)
    - `slf4j-simple-1.7.32.jar` (15 KB)
    - **Total JARs: ~3.2 MB**
  - Python: jpype1, cbor2, opencv, pillow, scikit-learn (só `DBSCAN`).
- **Custo aproximado:** ~200–500 ms por dedo (dominado pelo bootstrap da JVM no primeiro dedo; depois ~50–150 ms cada). **Bootstrap da JVM é caro** (~1–2s) — o processo CLI tem que viver o suficiente para amortizar.
- **Bugs/limitações:**
  - `analyze_minutiae_from_image` engole qualquer exceção e retorna 0. Bom para robustez, ruim para diagnóstico.
  - O cálculo de minutiae passa pela conversão `Image.fromarray → pil_img.tobytes()` antes de entrar no SourceAFIS — se a imagem não for grayscale puro, há conversão silenciosa.
  - **NÃO USAR JPype no .NET:** existe **SourceAFIS .NET nativo** (NuGet `SourceAFIS`), API idêntica. Em vez de porte via CLI, vale tentar primeiro uma DLL .NET pura que chama `SourceAFIS` direto. Se isso funcionar, **`MIcutMinutiaeCount` é a única feature deste inventário que pode ser empacotada como DLL pura, sem CLI.**
- **Arquivos-chave:**
  - `src/features/minutiae.py` (toda a lógica)
  - `bin/*.jar` (runtime Java)

### 2.2 `singularities_count` — quantidade de singularidades (cores/deltas)

- **Nome curto sugerido:** `MIcutSingularitiesCount`
- **O que mede:** quantos pontos singulares (core/delta) o SourceAFIS detectou. Vem do mesmo template gerado para `minutiae_count` (campo `singularities` do CBOR).
- **Faixa de saída:** inteiro 0–~10 típico. Sugiro `ScoreMaximo = 20`.
- **Entrada exigida:** mesma de `minutiae_count` (o template é o mesmo objeto).
- **Dependências runtime:** mesmas do `minutiae_count`.
- **Custo aproximado:** **custo zero adicional** se calculado junto com `minutiae_count` (sai do mesmo CBOR). Calcular separadamente = mesmo custo do `minutiae_count`.
- **Bugs/limitações:** o significado físico de "singularidade alta" é ambíguo — uma digital com 0 singularidades pode ser ruim OU pode ser um arco perfeito sem core. Não correlaciona necessariamente com qualidade. **Considerar incluir mais como metadado do que como score de qualidade**.
- **Arquivos-chave:** mesmos da §2.1.

### 2.3 `cluster_count` — número de clusters de minúcias após DBSCAN+merge

- **Nome curto sugerido:** `MIcutClusterCount`
- **O que mede:** depois de aplicar DBSCAN (eps=50, min_samples=5) sobre coordenadas das minúcias, filtrar clusters pequenos (<10% do total) e fundir clusters cujos centróides distam <100 px, conta os componentes resultantes. Um dedo "saudável" deveria produzir **1 cluster grande e contíguo**; >1 sugere "ilhas" de minúcias (digital fragmentada por ruído/máscara).
- **Faixa de saída:** inteiro 0–~5. Sugiro `ScoreMaximo = 10`.
- **Entrada exigida:** mesma de `minutiae_count`.
- **Dependências runtime:** mesmas + `sklearn.cluster.DBSCAN`. Em .NET: `Accord.MachineLearning` ou implementação manual (~30 linhas).
- **Custo aproximado:** custo zero adicional sobre `minutiae_count`.
- **Bugs/limitações:**
  - **Score "menor = melhor":** semanticamente, 1 cluster é ótimo e 0 ou >1 são ruins. Isso quebra a regra "maior=melhor" do `IFingerQualityExtractor`. Sugiro **inverter** no porte: retornar algo tipo `max(0, 100 - 20*(cluster_count-1))` ou `cluster_count == 1 ? 100 : 0`. **Definir essa regra antes do porte.**
  - Hardcodes (`eps=50, min_samples=5, ratio=0.1, dist=100`) são para a resolução 500 DPI e para o tamanho típico do crop. Mudar tamanho do crop → recalibrar.
- **Arquivos-chave:** `src/features/minutiae.py` (`detect_clusters_and_singularities`).

### 2.4 `solidity` — razão área/convex-hull do contorno da ROI

- **Nome curto sugerido:** `MIcutSolidity`
- **O que mede:** quão "preenchida" é a forma da digital. 1.0 = forma convexa perfeita; valores baixos indicam contorno irregular/recortado.
- **Faixa de saída:** float ∈ [0, 1]. **Precisa quantização para int.** Sugiro `int(round(solidity * 100))`, com `ScoreMaximo = 100`.
- **Entrada exigida:** **máscara ROI binária** (do dedo). **Não funciona sem máscara** — `analyze_shape(None)` ou `analyze_shape(mask_vazia)` retorna 0.
- **Dependências runtime:** apenas OpenCV (`findContours`, `approxPolyDP`, `convexHull`, `contourArea`, `boundingRect`). 1:1 em OpenCvSharp/Emgu.
- **Custo aproximado:** <5 ms.
- **Bugs/limitações:**
  - **Dependência crítica:** sem máscara confiável, a feature não tem sinal. No coletor, **o BMP que chega não vem com máscara** — vai ser preciso gerar a máscara dentro do extrator. A função `remove_lines_keep_fingerprints` em `ml_segmentation.py` é o padrão atual; o fallback Otsu de `texture.py` é mais simples mas menos preciso. **Esse é o ponto de decisão mais importante do porte** — ver §4.
  - Aproximação por `approxPolyDP(epsilon=0.01*arcLength)` reduz pontos e acelera, mas pode amaciar reentrâncias relevantes. Comportamento herdado, manter.
- **Arquivos-chave:** `src/features/shape.py`.

### 2.5 `coverage` — razão área/bounding-rect do contorno da ROI

- **Nome curto sugerido:** `MIcutCoverage`
- **O que mede:** quão bem a digital preenche seu próprio bounding box axial. Valores altos = digital alongada/elíptica preenchendo o retângulo; baixos = digital em diagonal ou com muita área vazia no retângulo.
- **Faixa de saída:** float ∈ [0, 1]. Quantizar `int(round(coverage * 100))`. `ScoreMaximo = 100`.
- **Entrada exigida:** mesma de `solidity` (mesma máscara, mesmo contorno).
- **Dependências runtime:** mesmas de `solidity`.
- **Custo aproximado:** custo zero adicional sobre `solidity`.
- **Bugs/limitações:**
  - Mesma dependência crítica da máscara.
  - `coverage` é função do **alinhamento** do dedo no crop: um dedo na diagonal terá `coverage` baixo independentemente da qualidade. **Pode não ser bom proxy de qualidade**. Avaliar com dados reais antes de tratar como score.
- **Arquivos-chave:** `src/features/shape.py`.

### 2.6 `sharpness` — variância do Laplacian (blur detection)

- **Nome curto sugerido:** `MIcutSharpness`
- **O que mede:** variância da segunda derivada (Laplacian) sobre o BMP em grayscale. Métrica clássica de detecção de borrado (Pech-Pacheco 2000). Maior = mais nitidez.
- **Faixa de saída:** float ≥ 0, **unbounded em teoria**. Valores observados para digital nítida a 500 DPI: ~500–3000. Para borrada: <100. Sugiro:
  - Quantização: `int(min(round(sharpness), 5000))`
  - `ScoreMaximo = 5000` (calibrar com amostra real do banco antes de definir thresholds).
- **Entrada exigida:** grayscale do dedo. **Não usa máscara ROI** — calcula sobre a imagem inteira (incluindo bordas que podem inflar a variância). Isso é uma **falha do código atual**: bordas pretas/brancas do crop inflam o Laplacian.
- **Dependências runtime:** apenas OpenCV (`Laplacian`). 1:1 em OpenCvSharp.
- **Custo aproximado:** <10 ms.
- **Bugs/limitações:**
  - **Sensível a borda:** ver acima. Se for portar como-está, fica fiel ao Python; se for "corrigir", aplicar máscara antes (calcular `lap.var()` só dentro da ROI). Decidir antes do porte.
  - Dependência forte de iluminação e do contraste do scanner. Pode variar mais entre máquinas do que entre dedos da mesma pessoa.
- **Arquivos-chave:** `src/features/texture.py` (linhas 84–85).

### 2.7 `orientation_std` — desvio-padrão circular do campo de orientação (graus)

- **Nome curto sugerido:** `MIcutOrientationStd`
- **O que mede:** quanto a orientação local das cristas varia entre blocos 16×16 da ROI. **Score menor = melhor** (digital com campo de orientação coerente; alto = caótico/borrado).
- **Faixa de saída:** float ∈ [0, ~103.9°] (limite teórico = `π/√3 · 180/π ≈ 103.9°` quando `R=0`). Quantizar: `int(round(orientation_std * 10))` → 0..1039. **OU inverter** para "maior=melhor": `int(round((104 - orientation_std) * 10))`. **Definir antes do porte.** Sugiro inverter — fica consistente com a regra.
- **Entrada exigida:** grayscale + máscara ROI binária. **Tem fallback Otsu se máscara ausente** (`texture.py` linhas 88–97), mas a função interna `block_orientation_std` exige `roi_mask` não-nulo (a chamada raiz aceita None e gera o fallback).
- **Dependências runtime:** OpenCV (`Sobel`, `GaussianBlur` no fallback, `threshold` Otsu, `morphologyEx`). 1:1 em OpenCvSharp.
- **Custo aproximado:** ~20–40 ms (loop block-by-block em Python — em C# será mais rápido).
- **Bugs/limitações:**
  - Usa estatística circular dobrando o ângulo (`2·θ`), correto para cristas π-periódicas. **Não substituir por `std` aritmético** no porte.
  - Hardcode `block=16` e threshold de 30% da ROI por bloco.
- **Arquivos-chave:** `src/features/texture.py` (`block_orientation_std`, linhas 30–56).

### 2.8 `contrast` — desvio-padrão dos níveis de cinza (RMS)

- **Nome curto sugerido:** `MIcutContrast`
- **O que mede:** `cv2.meanStdDev(gray)[1]` sobre o BMP inteiro. Maior = mais contraste entre cristas e vales.
- **Faixa de saída:** float ∈ [0, ~128]. Quantizar: `int(round(contrast))`. `ScoreMaximo = 128`.
- **Entrada exigida:** grayscale. **Não usa máscara** — calcula sobre o crop todo, inclusive bordas.
- **Dependências runtime:** OpenCV (`meanStdDev`). Trivial.
- **Custo aproximado:** <2 ms.
- **Bugs/limitações:**
  - Igual ao `sharpness`: inflado por bordas pretas/brancas. Se quiser ser fiel ao código atual, não aplicar máscara.
  - Métrica muito simples — pode não distinguir bem digital "boa" de digital "muito escurecida" (ambas têm std alto).
- **Arquivos-chave:** `src/features/texture.py` (linhas 103–104).

### 2.9 `ridge_consistency` — consistência da frequência das cristas

- **Nome curto sugerido:** `MIcutRidgeConsistency`
- **O que mede:** mede a homogeneidade da frequência local das cristas. Para cada bloco 32×32 dentro da ROI, calcula a orientação local, rotaciona o bloco para alinhar cristas verticalmente, projeta em 1D, faz FFT, pega o pico no intervalo `(1, block_size/2)`, valida wavelength ∈ (4, 20) px, acumula. Score final = `exp(-50 · std(frequências))`. Score = 1.0 → todas as cristas têm exatamente a mesma frequência.
- **Faixa de saída:** float ∈ (0, 1]. Quantizar: `int(round(ridge_consistency * 100))`. `ScoreMaximo = 100`.
- **Entrada exigida:** grayscale + máscara ROI (opcional — sem máscara, ainda funciona mas analisa blocos fora do dedo também).
- **Dependências runtime:** OpenCV (`Sobel`, `GaussianBlur`, `getRotationMatrix2D`, `warpAffine`) + `numpy.fft.fft` 1D.
- **Custo aproximado:** ~50–150 ms (vários blocos × FFT 1D em Python).
- **Bugs/limitações:**
  - Filtro de wavelength `(4, 20)` é calibrado para 500 DPI — coletor já usa BMPs 500 DPI da Dermalog, então OK.
  - FFT 1D em block_size=32 é grosseira; `peak_index` em `[1, 15]` é só 15 valores. Resolução de frequência limitada.
- **Arquivos-chave:** `src/features/frequency.py`.

---

## 3. Features descartadas (não são "features por dedo")

Para registro, descrevo o que NÃO entra no porte:

- **`run_stage_1_ml_segmentation` (Faster R-CNN, `best_detector_model_v2.pth`, 158 MB):** serve para detectar dedos dentro de uma "coluna" de 5 dedos concatenados. O coletor já tem o BMP de cada dedo separadamente (a Dermalog grava 1 BMP por dedo). **Toda a etapa de segmentação ML é dispensável no coletor.** Isso elimina a dependência mais pesada do projeto (PyTorch + 166 MB de modelo + ONNX).
- **`merge_and_rotate_fingerprints`:** idem — só serve para a entrada "5 dedos por imagem".
- **`run_stage_2_nfiq2` (NFIQ2.exe):** o coletor **já tem** integração NFIQ2 (extrator `NFIQ2` próprio). Não duplicar.
- **`remove_lines_keep_fingerprints`:** é a fábrica de máscara ROI. Não é uma feature em si, mas **é input crítico** de `solidity`, `coverage` e `orientation_std`. Discutido em §4.

---

## 4. Decisão crítica: como gerar a máscara ROI no coletor?

Três features (`solidity`, `coverage`, `orientation_std`) dependem da máscara ROI. No `micut_biometry` original, a máscara é subproduto da segmentação ML. No coletor, **não há ML**. Opções:

1. **Portar `remove_lines_keep_fingerprints` para o extrator** (adaptive threshold + remoção morfológica de linhas + dilatação). 1:1 em OpenCvSharp, ~30 linhas. **Recomendado.** Custo: ~10–20 ms.
2. **Usar fallback Otsu** (já existe em `texture.py` linhas 88–97). Mais simples, ~5 ms, mas menos preciso.
3. **Pular features que dependem de máscara.** Reduz inventário de 9 → 6 features.

**Recomendação:** opção 1. A função é pequena, determinística, sem ML, e replica fielmente o comportamento do pipeline atual.

---

## 5. Decisão crítica: como integrar SourceAFIS no .NET?

`minutiae_count`, `singularities_count`, `cluster_count` dependem de SourceAFIS.

- **No `micut_biometry`:** JPype invoca SourceAFIS-Java.
- **No .NET:** existe **`SourceAFIS` NuGet oficial** (.NET port mantido pelo mesmo autor — Robert Važan). API e formato CBOR idênticos. **Sem JVM, sem JPype, sem Java.**

**Recomendação:** usar o NuGet `SourceAFIS` diretamente. Isso transforma as 3 features de minúcias de "subprocess Python+Java" para **"DLL pura .NET"**. Elimina toda a dependência Java do inventário (os ~3.2 MB de JARs ficam fora). Para CBOR: `System.Formats.Cbor` (built-in .NET 5+) ou `PeterO.Cbor`. Para DBSCAN: implementação manual (~30 linhas) ou `Accord.MachineLearning`.

---

## 6. Decisão crítica: empacotamento (DLL pura vs. CLI Python)

Aplicando §4 (porte de `remove_lines_keep_fingerprints`) e §5 (SourceAFIS .NET):

- **Todas as 9 features podem ser portadas como DLL .NET pura**, sem Python e sem Java.
- Dependências NuGet finais: `SourceAFIS`, `OpenCvSharp4` + `OpenCvSharp4.runtime.win` (32-bit), `MathNet.Numerics` (para FFT 1D ou usar `Cv2.Dft`), opcional `Accord.MachineLearning`.
- Tamanho estimado do bundle: ~40 MB (OpenCvSharp nativo domina).

Isso **dispensa completamente o padrão "CLI subprocess"** descrito no prompt original. O Apêndice A do prompt fica reservado caso uma feature futura realmente precise de Python/Java.

---

## 7. Prioridade sugerida de porte

Critérios usados: (a) custo de implementação, (b) sinal complementar a FC3+NFIQ2 já existentes, (c) confiança na métrica.

| Prioridade | Feature | Justificativa |
|---|---|---|
| **1** | `MIcutMinutiaeCount` | Métrica clássica, robusta, fácil de interpretar. SourceAFIS .NET é maduro. Complementa NFIQ2 (que usa minúcias internamente mas não as expõe). |
| **2** | `MIcutRidgeConsistency` | Sinal forte e ortogonal ao FC3/NFIQ2: mede coerência de frequência, raramente avaliado por motores clássicos. Custo médio. |
| **3** | `MIcutOrientationStd` | Mesmo argumento do (2) na dimensão "coerência de orientação". Útil em conjunto com `RidgeConsistency`. |
| **4** | `MIcutSolidity` | Útil para detectar dedos cortados/parciais. Depende da qualidade da máscara ROI. |
| **5** | `MIcutCoverage` | Útil mas potencialmente ruidoso (sensível a alinhamento do dedo no crop). |
| **6** | `MIcutSharpness` | Métrica simples e barata. Cuidado com inflação por bordas (corrigir? ou manter fiel ao Python?). |
| **7** | `MIcutContrast` | Mesma observação do `Sharpness`. Métrica menos discriminante isoladamente. |
| **8** | `MIcutSingularitiesCount` | Mais metadado que score de qualidade. Útil para classificar tipo de digital (arco/laço/verticilo), pouco para qualidade. |
| **9** | `MIcutClusterCount` | Score "menor=melhor", precisa inversão. Hardcodes calibrados para crops do `micut_biometry` — pode precisar recalibrar para crops do coletor. |

**Recomendação operacional:** começar pelas 3 primeiras (minutiae + 2 de coerência) — isso dá 3 colunas novas em `FRC.NOVO` com sinal independente. As demais entram em batches subsequentes conforme análise estatística dos resultados.

---

## 8. Pendências / dúvidas a confirmar antes do porte

1. **Tamanho típico do BMP do coletor** (largura × altura em pixels). O `micut_biometry` opera sobre crops com largura variável + altura ~1000 px depois do `cv2.rotate`. Os BMPs da Dermalog têm tamanho conhecido (qual?). Isso afeta as constantes hardcoded (DBSCAN `eps=50`, `min_centroid_dist=100`, wavelength `(4, 20)`).
2. **DPI dos BMPs do coletor.** O SourceAFIS é muito sensível a isso. Se a Dermalog grava em DPI diferente de 500, ou se o DPI não está no header BMP, o template muda. Confirmar.
3. **Política para scores "menor=melhor" (`orientation_std`, `cluster_count`):** inverter no porte ou expor como-está? O contrato `IFingerQualityExtractor` diz "maior=melhor", então inverter parece a escolha certa, mas vale alinhar.
4. **Política para features dependentes de máscara em dedos amputados:** o contrato já garante retorno 0 para `EhAmputado=true`. Não há ação adicional, mas confirmar que não há slot "parcial" (amputação parcial) que possa quebrar.
5. **Calibração de `ScoreMaximo`:** valores propostos acima são chutes informados. Sob calibração real, podem mudar. Recomendo rodar a primeira leva de coleta com `ScoreMaximo` generoso e ajustar `QualityThresholds` no `appsettings.json` depois.
6. **Tratamento de BMP corrompido / OpenCvSharp lança:** o contrato pede retorno 0 sem exceção em erro de imagem isolada. OK.

---

## 9. Arquivos extras necessários para a DLL .NET (estimativa)

Assumindo a recomendação de §6 (DLL pura, sem Python/Java):

- `OpenCvSharp4.runtime.win` (32-bit) — natives ~20 MB.
- `SourceAFIS` NuGet — managed only, sem natives extras.
- `MathNet.Numerics` — managed only.
- **Nenhum modelo, nenhum executável CLI, nenhum JAR, nenhum binário NIST.**

Os 158 MB do `best_detector_model_v2.pth` e os 38 MB do NFIQ2 **ficam fora** (já justificado em §3).

---

## 10. Resumo executivo

- **9 features candidatas**, todas portáveis para DLL .NET 8 x86 **pura** (sem Python/Java/subprocess) se aceitarmos: (a) portar `remove_lines_keep_fingerprints` para gerar máscara ROI no extrator; (b) usar SourceAFIS .NET no lugar de SourceAFIS-Java+JPype.
- **Bundle estimado:** ~40 MB (dominado por OpenCvSharp nativo 32-bit).
- **Não há necessidade de manter o Faster R-CNN nem o NFIQ2 do `micut_biometry`** — o primeiro porque o coletor não opera sobre colunas-decadatilares, o segundo porque já existe.
- **Maior risco do porte:** divergência sutil entre OpenCvSharp e OpenCV-Python em casos de borda (raro, mas existe — `approxPolyDP` por exemplo). Validar com testes lado-a-lado num BMP real.
- **Próximo passo:** aprovação deste inventário + decisões de §4, §5, §8.
