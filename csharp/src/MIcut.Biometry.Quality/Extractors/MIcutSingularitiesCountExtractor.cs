using System.Diagnostics;
using MIcut.Biometry.Quality.Internal;

namespace MIcut.Biometry.Quality.Extractors;

public sealed class MIcutSingularitiesCountExtractor : IMIcutQualityExtractor
{
    public string Nome => "MIcutSingularitiesCount";
    public int ScoreMaximo => 20;

    public int ExtrairQualidade(int posicaoGrid, byte[]? imagemBmpBytes, bool ehAmputado)
    {
        if (ehAmputado || imagemBmpBytes is null || imagemBmpBytes.Length == 0) return 0;
        try
        {
            var template = SourceAfisTemplateCache.Instance.GetOrCreate(imagemBmpBytes);
            if (template is null) return 0;
            return Math.Min(template.SingularitiesCount, ScoreMaximo);
        }
        catch (Exception ex) when (ex is not OutOfMemoryException and not StackOverflowException)
        {
            Trace.TraceWarning($"[{Nome}] grid={posicaoGrid}: {ex.Message}");
            return 0;
        }
    }
}
