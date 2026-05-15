namespace MIcut.Biometry.Quality.Internal;

internal static class DbscanSimple
{
    public const int Noise = -1;

    public static int[] Cluster(double[][] points, double eps, int minSamples)
    {
        int n = points.Length;
        var labels = new int[n];
        Array.Fill(labels, Noise);
        if (n == 0) return labels;

        double eps2 = eps * eps;
        var visited = new bool[n];
        int clusterId = 0;

        for (int i = 0; i < n; i++)
        {
            if (visited[i]) continue;
            visited[i] = true;

            var neighbors = RegionQuery(points, i, eps2);
            if (neighbors.Count < minSamples) continue;

            labels[i] = clusterId;
            var queue = new Queue<int>(neighbors);
            while (queue.Count > 0)
            {
                int q = queue.Dequeue();
                if (!visited[q])
                {
                    visited[q] = true;
                    var qNeighbors = RegionQuery(points, q, eps2);
                    if (qNeighbors.Count >= minSamples)
                    {
                        foreach (int k in qNeighbors) queue.Enqueue(k);
                    }
                }
                if (labels[q] == Noise) labels[q] = clusterId;
            }
            clusterId++;
        }

        return labels;
    }

    private static List<int> RegionQuery(double[][] points, int p, double eps2)
    {
        var result = new List<int>();
        int n = points.Length;
        double px = points[p][0];
        double py = points[p][1];
        for (int q = 0; q < n; q++)
        {
            double dx = px - points[q][0];
            double dy = py - points[q][1];
            if (dx * dx + dy * dy <= eps2) result.Add(q);
        }
        return result;
    }
}
