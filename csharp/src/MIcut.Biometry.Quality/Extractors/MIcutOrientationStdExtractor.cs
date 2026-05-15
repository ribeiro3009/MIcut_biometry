using System.Diagnostics;
using MIcut.Biometry.Quality.Internal;
using OpenCvSharp;

namespace MIcut.Biometry.Quality.Extractors;

public sealed class MIcutOrientationStdExtractor : IMIcutQualityExtractor
{
    public string Nome => "MIcutOrientationStd";
    public int ScoreMaximo => 1040;

    private const int BlockSize = 16;
    private const double RoiCoverageFraction = 0.3;

    public int ExtrairQualidade(int posicaoGrid, byte[]? imagemBmpBytes, bool ehAmputado)
    {
        if (ehAmputado || imagemBmpBytes is null || imagemBmpBytes.Length == 0) return 0;
        try
        {
            var artifacts = BmpArtifactsCache.Instance.GetOrCreate(imagemBmpBytes);
            if (artifacts is null) return 0;

            using var gray = artifacts.NewGrayMat();
            using var mask = artifacts.NewMaskMat();

            double stdDeg = ComputeBlockOrientationStdDegrees(gray, mask);
            int rawScore = (int)Math.Round(stdDeg * 10.0, MidpointRounding.AwayFromZero);
            rawScore = Math.Clamp(rawScore, 0, ScoreMaximo);
            return ScoreMaximo - rawScore;
        }
        catch (Exception ex) when (ex is not OutOfMemoryException and not StackOverflowException)
        {
            Trace.TraceWarning($"[{Nome}] grid={posicaoGrid}: {ex.Message}");
            return 0;
        }
    }

    private static double ComputeBlockOrientationStdDegrees(Mat gray, Mat mask)
    {
        int H = gray.Rows;
        int W = gray.Cols;
        var angles = new List<double>();
        double blockArea = BlockSize * BlockSize;
        double roiThreshold = blockArea * RoiCoverageFraction;

        using var gxFull = new Mat();
        using var gyFull = new Mat();
        Cv2.Sobel(gray, gxFull, MatType.CV_32F, 1, 0, ksize: 3);
        Cv2.Sobel(gray, gyFull, MatType.CV_32F, 0, 1, ksize: 3);

        var gxIdx = gxFull.GetGenericIndexer<float>();
        var gyIdx = gyFull.GetGenericIndexer<float>();
        var maskIdx = mask.GetGenericIndexer<byte>();

        for (int y = 0; y + BlockSize <= H; y += BlockSize)
        {
            for (int x = 0; x + BlockSize <= W; x += BlockSize)
            {
                long roiSum = 0;
                for (int j = 0; j < BlockSize; j++)
                    for (int i = 0; i < BlockSize; i++)
                        if (maskIdx[y + j, x + i] != 0) roiSum++;

                if (roiSum < roiThreshold) continue;

                double vx = 0.0;
                double vy = 0.0;
                for (int j = 0; j < BlockSize; j++)
                {
                    for (int i = 0; i < BlockSize; i++)
                    {
                        double gx = gxIdx[y + j, x + i];
                        double gy = gyIdx[y + j, x + i];
                        vx += 2.0 * gx * gy;
                        vy += gx * gx - gy * gy;
                    }
                }
                double theta = 0.5 * Math.Atan2(vx, vy);
                angles.Add(theta);
            }
        }

        if (angles.Count == 0) return 0.0;

        double cosSum = 0.0;
        double sinSum = 0.0;
        foreach (double t in angles)
        {
            cosSum += Math.Cos(2.0 * t);
            sinSum += Math.Sin(2.0 * t);
        }
        double meanC = cosSum / angles.Count;
        double meanS = sinSum / angles.Count;
        double R = Math.Sqrt(meanC * meanC + meanS * meanS);

        double circStd = R > 0
            ? Math.Sqrt(-2.0 * Math.Log(R))
            : Math.PI / Math.Sqrt(3.0);

        return circStd * (180.0 / Math.PI);
    }
}
