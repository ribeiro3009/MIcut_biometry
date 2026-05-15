using System.Diagnostics;
using MIcut.Biometry.Quality.Internal;

namespace MIcut.Biometry.Quality.Extractors;

public sealed class MIcutCoverageExtractor : IMIcutQualityExtractor
{
    public string Nome => "MIcutCoverage";
    public int ScoreMaximo => 100;

    public int ExtrairQualidade(int posicaoGrid, byte[]? imagemBmpBytes, bool ehAmputado)
    {
        if (ehAmputado || imagemBmpBytes is null || imagemBmpBytes.Length == 0) return 0;
        try
        {
            var artifacts = BmpArtifactsCache.Instance.GetOrCreate(imagemBmpBytes);
            if (artifacts is null) return 0;
            using var mask = artifacts.NewMaskMat();
            var result = ShapeMetrics.Compute(mask);
            int score = (int)Math.Round(result.Coverage * 100, MidpointRounding.AwayFromZero);
            return Math.Clamp(score, 0, ScoreMaximo);
        }
        catch (Exception ex) when (ex is not OutOfMemoryException and not StackOverflowException)
        {
            Trace.TraceWarning($"[{Nome}] grid={posicaoGrid}: {ex.Message}");
            return 0;
        }
    }
}
