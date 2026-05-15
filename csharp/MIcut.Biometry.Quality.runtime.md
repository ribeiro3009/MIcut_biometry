# MIcut.Biometry.Quality.runtime.md

Lista exata de dependências e arquivos extras que viajam com a DLL.

## NuGet packages (versões pinadas)

| Package | Version | Por que |
|---|---|---|
| `OpenCvSharp4` | `4.9.0.20240103` | Operações de imagem (Sobel, Laplacian, MorphologyEx, FindContours, ConvexHull, GaussianBlur, WarpAffine, AdaptiveThreshold, MeanStdDev, ImDecode). |
| `OpenCvSharp4.runtime.win` | `4.9.0.20240103` | Binários nativos x86+x64 para Windows. |
| `SourceAFIS` | `3.14.0` | Geração do template de minúcias/singularidades. **Versão menor do que a usada no Python** (que é Java `sourceafis-3.18.1.jar`). Ver §"Caveat SourceAFIS" abaixo. |
| `System.Formats.Cbor` | `8.0.0` | Parser CBOR para extrair `positionsX`/`positionsY`/`singularities` do template SourceAFIS. |

Restore: `dotnet restore` na raiz `csharp/` (a partir do `MIcut.Biometry.Quality.sln`).

## Caveat SourceAFIS — versão 3.14 vs 3.18

- O `micut_biometry` (Python) usa o JAR Java `sourceafis-3.18.1.jar`.
- O NuGet `SourceAFIS` (porte .NET oficial mantido por Robert Važan, mesmo autor) só vai até `3.14.0` (verificado em `https://api.nuget.org/v3-flatcontainer/sourceafis/index.json`).
- O formato CBOR do template é estável ao longo da série 3.x — os campos `positionsX`, `positionsY`, `singularities` existem nas duas versões.
- **Impacto esperado:** contagem de minúcias e singularidades pode divergir ligeiramente entre Python (Java 3.18) e .NET (3.14). Em testes informais a variação fica em poucos pontos percentuais por dedo, raramente alterando a ordenação de qualidade entre dedos da mesma pessoa. Validar com amostra real antes de calibrar `QualityThresholds` no coletor.

## Arquivos no output (`bin/x86/Release/net8.0-windows/`)

Managed (gerenciados):

| Arquivo | Tamanho aprox. | Origem |
|---|---|---|
| `MIcut.Biometry.Quality.dll` | ~30 KB | Este projeto |
| `OpenCvSharp.dll` | ~950 KB | NuGet OpenCvSharp4 |
| `SourceAFIS.dll` | ~90 KB | NuGet SourceAFIS |
| `System.Formats.Cbor.dll` | ~90 KB | NuGet System.Formats.Cbor |

Nativos (em `runtimes/win-x86/native/`):

| Arquivo | Tamanho aprox. | Origem |
|---|---|---|
| `OpenCvSharpExtern.dll` | ~43 MB | OpenCvSharp4.runtime.win |
| `opencv_videoio_ffmpeg490.dll` | ~23 MB | OpenCvSharp4.runtime.win |

**Total estimado: ~67 MB**, dominado pelos nativos OpenCV. O `opencv_videoio_ffmpeg490.dll` é codec de vídeo — **não é usado por este projeto** (nenhuma chamada `VideoCapture`/`VideoWriter`). Pode ser excluído pelo consumidor para reduzir o bundle a ~44 MB, removendo o arquivo manualmente do diretório de saída. Não removemos via `<ExcludeAssets>` no csproj porque isso quebra a topologia padrão do `runtime.win` em consumidores que adicionarem chamadas de I/O de vídeo no futuro.

A estimativa original de ~40 MB no `INVENTARIO_FEATURES.md` (§6) considerava só o `OpenCvSharpExtern.dll`. Confirmado: sem o ffmpeg o bundle é de ~44 MB; com ele, ~67 MB.

## Sem dependências externas em runtime

- Nenhum executável CLI (Python, Java, NFIQ2.exe).
- Nenhum modelo (Faster R-CNN `.pth`, NFIQ2 `.yaml`).
- Nenhum acesso à internet.
- Nenhum hardware dongle.

## Decisão de FFT 1D

Implementação manual em `Internal/Dft1D.cs` (DFT direto O(N²) com `N=32` — ~1024 operações por bloco, trivial). Não trouxemos `MathNet.Numerics` (overhead grande para uma única função de FFT 1D usada em um único extrator) nem usamos `Cv2.Dft` (layout de saída CCS-packed exige unpacking manual, perde a vantagem). O custo total dominante do `MIcutRidgeConsistencyExtractor` é a `WarpAffine` por bloco, não a DFT.

## Compatibilidade x86

- `PlatformTarget = x86` em ambos os csproj.
- `OpenCvSharp4.runtime.win` traz natives x86 e x64 — o runtime seleciona o `win-x86` quando o processo é 32-bit, automaticamente.
- `SourceAFIS` é managed-only — não tem dependência de arquitetura.
- Para **executar `dotnet test`**, a máquina precisa do **.NET 8 Desktop Runtime x86** instalado (https://aka.ms/dotnet-core-applaunch?missing_runtime=true&arch=x86&rid=win10-x86). Em máquinas só com runtime x64 instalado, o build passa mas a execução dos testes falha com "hostfxr.dll could not be found". Em produção, o coletor já tem o runtime x86 instalado por outros motivos (FC3 ComInterop).

## Licenças

- **OpenCvSharp**: BSD-3-Clause (https://github.com/shimat/opencvsharp). Compatível com uso comercial.
- **OpenCV nativo (via OpenCvSharp.runtime.win)**: Apache 2.0 (a partir de OpenCV 4.5). Compatível com uso comercial.
- **SourceAFIS**: Apache 2.0 (https://sourceafis.machinezoo.com/). Compatível com uso comercial.
- **System.Formats.Cbor**: MIT (Microsoft .NET runtime). Compatível.

Nenhum dos pacotes exige atribuição visível em produto final, mas é boa prática manter um `THIRD_PARTY_NOTICES.txt` no coletor com os textos de licença completos.
