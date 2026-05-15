using System.Diagnostics;
using MIcut.Biometry.Quality.Internal;
using OpenCvSharp;

namespace MIcut.Biometry.Quality.Extractors;

public sealed class MIcutSharpnessExtractor : IMIcutQualityExtractor
{
    public string Nome => "MIcutSharpness";
    public int ScoreMaximo => 5000;

    public int ExtrairQualidade(int posicaoGrid, byte[]? imagemBmpBytes, bool ehAmputado)
    {
        if (ehAmputado || imagemBmpBytes is null || imagemBmpBytes.Length == 0) return 0;
        try
        {
            var artifacts = BmpArtifactsCache.Instance.GetOrCreate(imagemBmpBytes);
            if (artifacts is null) return 0;

            using var gray = artifacts.NewGrayMat();
            using var mask = artifacts.NewMaskMat();
            using var lap = new Mat();
            Cv2.Laplacian(gray, lap, MatType.CV_64F);

            double variance = MaskedVariance(lap, mask);
            int score = (int)Math.Round(variance, MidpointRounding.AwayFromZero);
            return Math.Clamp(score, 0, ScoreMaximo);
        }
        catch (Exception ex) when (ex is not OutOfMemoryException and not StackOverflowException)
        {
            Trace.TraceWarning($"[{Nome}] grid={posicaoGrid}: {ex.Message}");
            return 0;
        }
    }

    private static double MaskedVariance(Mat lap64, Mat mask)
    {
        int rows = lap64.Rows;
        int cols = lap64.Cols;

        double sum = 0.0;
        long count = 0;
        var lapIdx = lap64.GetGenericIndexer<double>();
        var maskIdx = mask.GetGenericIndexer<byte>();

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                if (maskIdx[r, c] == 0) continue;
                sum += lapIdx[r, c];
                count++;
            }
        }
        if (count < 2) return 0.0;
        double mean = sum / count;

        double sumSq = 0.0;
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                if (maskIdx[r, c] == 0) continue;
                double d = lapIdx[r, c] - mean;
                sumSq += d * d;
            }
        }
        return sumSq / count;
    }
}
