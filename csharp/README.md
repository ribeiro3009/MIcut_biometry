# MIcut.Biometry.Quality

DLL .NET 8 / x86 que expõe 9 features de qualidade de impressão digital portadas do projeto Python `micut_biometry` para consumo pelo `ColetorQualidades.WinForms`.

## Entregáveis

Após `dotnet build -c Release`, o output está em:

```
csharp/src/MIcut.Biometry.Quality/bin/x86/Release/net8.0-windows/
├── MIcut.Biometry.Quality.dll        ← a DLL principal
├── MIcut.Biometry.Quality.pdb
├── OpenCvSharp.dll
├── SourceAFIS.dll
├── System.Formats.Cbor.dll
└── runtimes/win-x86/native/
    ├── OpenCvSharpExtern.dll
    └── opencv_videoio_ffmpeg490.dll  ← opcional, podem remover
```

Lista completa de arquivos e tamanhos: ver [`MIcut.Biometry.Quality.runtime.md`](MIcut.Biometry.Quality.runtime.md).

Como integrar no consumidor: ver [`../INTEGRACAO_NO_COLETOR.md`](../INTEGRACAO_NO_COLETOR.md).

Inventário das 9 features: ver [`../INVENTARIO_FEATURES.md`](../INVENTARIO_FEATURES.md).

## Estrutura

```
csharp/
├── MIcut.Biometry.Quality.sln
├── README.md                          (este arquivo)
├── MIcut.Biometry.Quality.runtime.md
├── src/MIcut.Biometry.Quality/
│   ├── MIcut.Biometry.Quality.csproj
│   ├── IMIcutQualityExtractor.cs
│   ├── MIcutBiometryQualityExtensions.cs
│   ├── Extractors/                    (9 classes públicas)
│   └── Internal/                      (helpers: BmpDecoder, RoiMaskBuilder,
│                                       SourceAfisTemplateCache, BmpArtifactsCache,
│                                       TemplateData, ShapeMetrics, DbscanSimple, Dft1D)
└── tests/MIcut.Biometry.Quality.Tests/
    ├── MIcut.Biometry.Quality.Tests.csproj
    ├── ExtractorContractTests.cs      (contrato: amp/null/empty/garbage → 0)
    ├── BomVsRuimOrderingTests.cs      (bom > ruim em extratores baseados em máscara)
    └── Resources/finger_sample.bmp    (BMP real, embedded)
```

## Build

```powershell
cd csharp
dotnet build -c Release
```

Esperado: `Build succeeded. 0 Warning(s). 0 Error(s).`

## Testes

```powershell
cd csharp
dotnet test -c Release
```

**Pré-requisito:** .NET 8 Desktop Runtime **x86** instalado (não basta o x64). Download:
https://aka.ms/dotnet-core-applaunch?missing_runtime=true&arch=x86&rid=win10-x86

Em máquinas só com runtime x64, o build passa mas `dotnet test` aborta com `hostfxr.dll could not be found`. A máquina-alvo do coletor já tem o runtime x86 instalado por causa de outras dependências do projeto (FC3 ComInterop).

Os testes cobrem:
- **Contrato:** cada um dos 9 extratores retorna `0` para amputado/null/empty/garbage e respeita `[0, ScoreMaximo]` em BMP real.
- **Ordenação bom > ruim:** o BMP de teste é embedded; a versão "ruim" é gerada por `GaussianBlur` agressivo (kernel 51×51) sobre o mesmo BMP. Validado para `Sharpness`, `RidgeConsistency`, `OrientationStd`, `Contrast`, `MinutiaeCount`.

## Caveats

- **Plataforma:** `PlatformTarget = x86` por contrato com o coletor (FC3 é COM 32-bit). Build e execução exigem ambiente x86 .NET 8.
- **SourceAFIS versão:** o NuGet só vai até `3.14.0`; o `micut_biometry` Python usa Java `3.18.1`. Pequena divergência possível em contagens de minúcias/singularidades. Detalhes em `MIcut.Biometry.Quality.runtime.md`.
- **Divergência intencional Python vs C#:** `MIcutSharpness` e `MIcutContrast` aplicam máscara ROI; o Python não. Documentado em `INTEGRACAO_NO_COLETOR.md` §4.
- **Caches internos:** `SourceAfisTemplateCache` (32 entries) e `BmpArtifactsCache` (16 entries) evitam recomputar template SourceAFIS e máscara ROI quando os 9 extratores são chamados em sequência para o mesmo BMP. Thread-safe via `lock`.
