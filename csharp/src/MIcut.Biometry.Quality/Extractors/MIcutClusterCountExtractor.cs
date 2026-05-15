using System.Diagnostics;
using MIcut.Biometry.Quality.Internal;

namespace MIcut.Biometry.Quality.Extractors;

public sealed class MIcutClusterCountExtractor : IMIcutQualityExtractor
{
    public string Nome => "MIcutClusterCount";
    public int ScoreMaximo => 100;

    private const double Eps = 50.0;
    private const int MinSamples = 5;
    private const double ClusterSizeRatio = 0.1;
    private const double MinCentroidDist = 100.0;

    public int ExtrairQualidade(int posicaoGrid, byte[]? imagemBmpBytes, bool ehAmputado)
    {
        if (ehAmputado || imagemBmpBytes is null || imagemBmpBytes.Length == 0) return 0;
        try
        {
            var template = SourceAfisTemplateCache.Instance.GetOrCreate(imagemBmpBytes);
            if (template is null) return 0;

            int rawClusters = CountClusters(template);
            return rawClusters switch
            {
                1 => 100,
                0 => 0,
                _ => Math.Max(0, 100 - 25 * (rawClusters - 1))
            };
        }
        catch (Exception ex) when (ex is not OutOfMemoryException and not StackOverflowException)
        {
            Trace.TraceWarning($"[{Nome}] grid={posicaoGrid}: {ex.Message}");
            return 0;
        }
    }

    private static int CountClusters(TemplateData t)
    {
        int n = t.MinutiaeCount;
        if (n == 0) return 0;
        if (n < MinSamples) return 1;

        var points = new double[n][];
        for (int i = 0; i < n; i++)
            points[i] = new double[] { t.PositionsX[i], t.PositionsY[i] };

        int[] labels = DbscanSimple.Cluster(points, Eps, MinSamples);

        var counts = new Dictionary<int, int>();
        int totalInClusters = 0;
        for (int i = 0; i < n; i++)
        {
            if (labels[i] < 0) continue;
            totalInClusters++;
            counts.TryGetValue(labels[i], out int c);
            counts[labels[i]] = c + 1;
        }
        if (totalInClusters == 0) return 0;

        double threshold = ClusterSizeRatio * totalInClusters;
        var largeClusterIds = counts.Where(kv => kv.Value >= threshold).Select(kv => kv.Key).ToArray();
        if (largeClusterIds.Length == 0) return 0;

        var centroids = new (double X, double Y)[largeClusterIds.Length];
        for (int c = 0; c < largeClusterIds.Length; c++)
        {
            int id = largeClusterIds[c];
            double sx = 0, sy = 0;
            int k = 0;
            for (int i = 0; i < n; i++)
            {
                if (labels[i] == id)
                {
                    sx += points[i][0];
                    sy += points[i][1];
                    k++;
                }
            }
            centroids[c] = (sx / k, sy / k);
        }

        var visited = new bool[centroids.Length];
        int components = 0;
        for (int i = 0; i < centroids.Length; i++)
        {
            if (visited[i]) continue;
            var stack = new Stack<int>();
            stack.Push(i);
            while (stack.Count > 0)
            {
                int j = stack.Pop();
                if (visited[j]) continue;
                visited[j] = true;
                for (int k = 0; k < centroids.Length; k++)
                {
                    if (visited[k]) continue;
                    double dx = centroids[j].X - centroids[k].X;
                    double dy = centroids[j].Y - centroids[k].Y;
                    if (Math.Sqrt(dx * dx + dy * dy) < MinCentroidDist) stack.Push(k);
                }
            }
            components++;
        }
        return components;
    }
}
