# INTEGRACAO_NO_COLETOR.md

Como consumir `MIcut.Biometry.Quality.dll` a partir do `ColetorQualidades.WinForms` (.NET 8 / x86).

## 1. Classes públicas expostas

Namespace: `MIcut.Biometry.Quality`.
Subnamespace: `MIcut.Biometry.Quality.Extractors` (onde moram as 9 classes).

Todas implementam `IMIcutQualityExtractor`:

```csharp
namespace MIcut.Biometry.Quality;

public interface IMIcutQualityExtractor
{
    int ExtrairQualidade(int posicaoGrid, byte[]? imagemBmpBytes, bool ehAmputado);
    int ScoreMaximo { get; }
    string Nome { get; }
}
```

| # | Classe | `Nome` | `ScoreMaximo` | Coluna sugerida em `FRC.NOVO` |
|---|---|---|---|---|
| 1 | `MIcutMinutiaeCountExtractor` | `MIcutMinutiaeCount` | 200 | `MICUT_MINUTIAE_COUNT` |
| 2 | `MIcutSingularitiesCountExtractor` | `MIcutSingularitiesCount` | 20 | `MICUT_SINGULARITIES_COUNT` |
| 3 | `MIcutClusterCountExtractor` | `MIcutClusterCount` | 100 | `MICUT_CLUSTER_COUNT` |
| 4 | `MIcutSolidityExtractor` | `MIcutSolidity` | 100 | `MICUT_SOLIDITY` |
| 5 | `MIcutCoverageExtractor` | `MIcutCoverage` | 100 | `MICUT_COVERAGE` |
| 6 | `MIcutSharpnessExtractor` | `MIcutSharpness` | 5000 | `MICUT_SHARPNESS` |
| 7 | `MIcutOrientationStdExtractor` | `MIcutOrientationStd` | 1040 | `MICUT_ORIENTATION_STD` |
| 8 | `MIcutContrastExtractor` | `MIcutContrast` | 128 | `MICUT_CONTRAST` |
| 9 | `MIcutRidgeConsistencyExtractor` | `MIcutRidgeConsistency` | 100 | `MICUT_RIDGE_CONSISTENCY` |

DDL sugerida para a expansão de `FRC.NOVO`:

```sql
ALTER TABLE FRC.NOVO ADD (
  MICUT_MINUTIAE_COUNT     VARCHAR2(300),
  MICUT_SINGULARITIES_COUNT VARCHAR2(300),
  MICUT_CLUSTER_COUNT      VARCHAR2(300),
  MICUT_SOLIDITY           VARCHAR2(300),
  MICUT_COVERAGE           VARCHAR2(300),
  MICUT_SHARPNESS          VARCHAR2(300),
  MICUT_ORIENTATION_STD    VARCHAR2(300),
  MICUT_CONTRAST           VARCHAR2(300),
  MICUT_RIDGE_CONSISTENCY  VARCHAR2(300)
);
```

Formato dos valores: `"q0;q1;q2;q3;q4;q5;q6;q7;q8;q9;"` — mesmo padrão dos campos `FC3`/`NFIQ2`.

## 2. Registro DI

Cada extrator é registrado como `Singleton`. O construtor não tem parâmetros — todo o estado caro (template SourceAFIS, máscara ROI) é gerenciado por caches internos thread-safe.

```csharp
using MIcut.Biometry.Quality;
using MIcut.Biometry.Quality.Extractors;

services.AddSingleton<IMIcutQualityExtractor, MIcutMinutiaeCountExtractor>();
services.AddSingleton<IMIcutQualityExtractor, MIcutSingularitiesCountExtractor>();
services.AddSingleton<IMIcutQualityExtractor, MIcutClusterCountExtractor>();
services.AddSingleton<IMIcutQualityExtractor, MIcutSolidityExtractor>();
services.AddSingleton<IMIcutQualityExtractor, MIcutCoverageExtractor>();
services.AddSingleton<IMIcutQualityExtractor, MIcutSharpnessExtractor>();
services.AddSingleton<IMIcutQualityExtractor, MIcutOrientationStdExtractor>();
services.AddSingleton<IMIcutQualityExtractor, MIcutContrastExtractor>();
services.AddSingleton<IMIcutQualityExtractor, MIcutRidgeConsistencyExtractor>();
```

E consume via `IEnumerable<IMIcutQualityExtractor>` no `ColetorService`.

Para enumerar programaticamente (sem injeção), há o helper estático:

```csharp
foreach (var extractor in MIcutBiometryQualityRegistry.CreateAll())
{
    int score = extractor.ExtrairQualidade(posicaoGrid: i, imagemBmpBytes: bmp, ehAmputado: amp);
    // ...
}
```

Esse helper é apenas conveniência para batch processing fora de DI — em produção use DI.

## 3. Arquivos para copiar ao output do consumidor

Ver `MIcut.Biometry.Quality.runtime.md` para a lista completa. Resumo:

```
MIcut.Biometry.Quality.dll
OpenCvSharp.dll
SourceAFIS.dll
System.Formats.Cbor.dll
runtimes/win-x86/native/OpenCvSharpExtern.dll
runtimes/win-x86/native/opencv_videoio_ffmpeg490.dll   <-- opcional, podem remover (~23 MB)
```

Bundle total: ~67 MB com ffmpeg, ~44 MB sem.

## 4. Divergência intencional com o pipeline Python

**`MIcutSharpness` e `MIcutContrast` aplicam a máscara ROI antes de calcular**, divergindo deliberadamente do `micut_biometry` em Python que calcula sobre o crop inteiro (inflando o resultado com bordas/fundo).

Justificativa: o objetivo do coletor é capturar sinal de qualidade utilizável, não reproduzir um bug do pipeline original. Se for desejado reproduzir os números do CSV do `micut_biometry` em validação cruzada, **não usar essas duas features** — usar diretamente o pipeline Python.

Documentado também em `MIcut.Biometry.Quality.runtime.md`.

## 5. Inversão de score (regra "menor = melhor" → "maior = melhor")

Duas features foram **invertidas** dentro do extrator para casar com a convenção `IFingerQualityExtractor` (maior = melhor):

- **`MIcutClusterCount`** — `cluster_count` cru: 1 cluster é ideal, 0 ou >1 é ruim. Regra de inversão:
  - 0 clusters → score 0
  - 1 cluster → score 100
  - 2 clusters → 75
  - 3 → 50
  - 4 → 25
  - ≥5 → 0
- **`MIcutOrientationStd`** — `orientation_std` cru ∈ [0, ~103.9°] (menor = mais coerente). Inversão: `score = 1040 - round(orientationStd * 10)`. Score=1040 quando perfeitamente coerente, score=0 quando totalmente caótico.

## 6. Garantias de runtime (do contrato)

- `ehAmputado == true` ou `imagemBmpBytes` null/vazio → retorno imediato 0 (sem chamadas custosas).
- BMP inválido ou corrompido → 0 (sem exceção propagada).
- DLL nativa OpenCvSharp ausente / NuGet mal-resolvido → exceção do tipo `DllNotFoundException` ocorre no **primeiro uso** (não no construtor) — não há `init()` proativo. Se quiser falhar cedo, chame `new MIcutContrastExtractor().ExtrairQualidade(0, new byte[]{}, false)` na inicialização do coletor; isso não força o carregamento dos natives. Para forçar, faça uma decodificação dummy via `OpenCvSharp.Cv2.ImDecode(new byte[]{0x42,0x4D}, ImreadModes.Grayscale)` no startup.
- Thread-safety: todos os 9 extratores são seguros para chamadas concorrentes — caches internos (`SourceAfisTemplateCache`, `BmpArtifactsCache`) usam `lock`.

## 7. Custo aproximado por dedo

Medições informais (BMP ~500 DPI, ~300×400 px, máquina desktop comum):

| Feature | Primeira chamada (cold cache) | Subsequentes (hit) |
|---|---|---|
| MinutiaeCount + Singularities + Cluster (compartilham template) | ~200 ms (SourceAFIS build) | < 5 ms cada (cache hit) |
| Solidity + Coverage (compartilham máscara) | ~25 ms (decode + mask build) | < 10 ms cada (cache hit) |
| Sharpness, Contrast, OrientationStd, RidgeConsistency | ~25 ms na primeira (mesma máscara), depois < 30 ms cada |

Total por dedo (9 features): **~300 ms na cold cache, ~50–100 ms total quando o mesmo BMP foi visto recentemente**. Como o coletor chama os 9 extratores em sequência para o mesmo BMP, o pior caso prático é ~300 ms/dedo.

10 dedos por PID × 50K PIDs = ~14 horas em serial, ou ~2 horas com `Parallel.ForEach` em 8 cores. Aceitável para coleta noturna.

## 8. Licenças

- **OpenCvSharp**: BSD-3-Clause
- **OpenCV nativo**: Apache 2.0
- **SourceAFIS**: Apache 2.0
- **System.Formats.Cbor**: MIT

Nenhum hardware dongle. Nenhum modelo restrito. Uso comercial liberado em todas as deps.

## 9. Estrutura recomendada no coletor

```csharp
public class FRCNovoRow {
    public long NU_PID { get; set; }
    public string? FC3 { get; set; }
    public string? NFIQ2 { get; set; }
    public string? MICUT_MINUTIAE_COUNT { get; set; }
    public string? MICUT_SINGULARITIES_COUNT { get; set; }
    public string? MICUT_CLUSTER_COUNT { get; set; }
    public string? MICUT_SOLIDITY { get; set; }
    public string? MICUT_COVERAGE { get; set; }
    public string? MICUT_SHARPNESS { get; set; }
    public string? MICUT_ORIENTATION_STD { get; set; }
    public string? MICUT_CONTRAST { get; set; }
    public string? MICUT_RIDGE_CONSISTENCY { get; set; }
    public string? MS_ERRO { get; set; }
}

// no ColetorService:
foreach (var extractor in _extractors) {
    var scores = new int[10];
    for (int i = 0; i < 10; i++) {
        bool amp = stDigitais[i] is 'A' or 'P';
        scores[i] = extractor.ExtrairQualidade(i, bmps[i], amp);
    }
    string concatenado = string.Concat(scores.Select(s => s + ";"));
    row.SetByName(extractor.Nome, concatenado);
}
```

## 10. Cuidados operacionais

- **Primeira execução é lenta**: o SourceAFIS faz JIT warm-up nas primeiras chamadas (modelo de minúcias). Conte ~1–2s de overhead no primeiro dedo de cada processo. Em loops longos esse custo se dilui.
- **Memória**: os caches internos têm limite de ~16 BMPs (BmpArtifactsCache) e ~32 templates (SourceAfisTemplateCache) — pico estimado <50 MB. Não precisa flush manual.
- **PDB**: o arquivo `MIcut.Biometry.Quality.pdb` é útil para stack traces em produção. Copiar junto.
