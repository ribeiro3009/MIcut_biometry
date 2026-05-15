using System.Diagnostics;
using MIcut.Biometry.Quality.Internal;
using OpenCvSharp;

namespace MIcut.Biometry.Quality.Extractors;

public sealed class MIcutRidgeConsistencyExtractor : IMIcutQualityExtractor
{
    public string Nome => "MIcutRidgeConsistency";
    public int ScoreMaximo => 100;

    private const int BlockSize = 32;
    private const double RoiCoverageFraction = 0.25;
    private const double MinWavelength = 4.0;
    private const double MaxWavelength = 20.0;
    private const double K = 50.0;

    public int ExtrairQualidade(int posicaoGrid, byte[]? imagemBmpBytes, bool ehAmputado)
    {
        if (ehAmputado || imagemBmpBytes is null || imagemBmpBytes.Length == 0) return 0;
        try
        {
            var artifacts = BmpArtifactsCache.Instance.GetOrCreate(imagemBmpBytes);
            if (artifacts is null) return 0;

            using var gray = artifacts.NewGrayMat();
            using var mask = artifacts.NewMaskMat();

            double consistency = ComputeRidgeConsistency(gray, mask);
            int score = (int)Math.Round(consistency * 100, MidpointRounding.AwayFromZero);
            return Math.Clamp(score, 0, ScoreMaximo);
        }
        catch (Exception ex) when (ex is not OutOfMemoryException and not StackOverflowException)
        {
            Trace.TraceWarning($"[{Nome}] grid={posicaoGrid}: {ex.Message}");
            return 0;
        }
    }

    private static double ComputeRidgeConsistency(Mat gray, Mat mask)
    {
        int H = gray.Rows;
        int W = gray.Cols;

        using var sobelX = new Mat();
        using var sobelY = new Mat();
        Cv2.Sobel(gray, sobelX, MatType.CV_64F, 1, 0, ksize: 5);
        Cv2.Sobel(gray, sobelY, MatType.CV_64F, 0, 1, ksize: 5);

        using var sx2 = sobelX.Mul(sobelX).ToMat();
        using var sy2 = sobelY.Mul(sobelY).ToMat();
        using var sxy = sobelX.Mul(sobelY).ToMat();

        using var gxx = new Mat();
        using var gyy = new Mat();
        using var gxy = new Mat();
        Cv2.GaussianBlur(sx2, gxx, new Size(5, 5), 0);
        Cv2.GaussianBlur(sy2, gyy, new Size(5, 5), 0);
        Cv2.GaussianBlur(sxy, gxy, new Size(5, 5), 0);

        var gxxIdx = gxx.GetGenericIndexer<double>();
        var gyyIdx = gyy.GetGenericIndexer<double>();
        var gxyIdx = gxy.GetGenericIndexer<double>();
        var maskIdx = mask.GetGenericIndexer<byte>();
        var grayIdx = gray.GetGenericIndexer<byte>();

        var frequencies = new List<double>();
        double[] projection = new double[BlockSize];
        double[] rotated = new double[BlockSize * BlockSize];

        double roiThreshold = RoiCoverageFraction;

        for (int r = 0; r + BlockSize <= H; r += BlockSize)
        {
            for (int c = 0; c + BlockSize <= W; c += BlockSize)
            {
                long maskCount = 0;
                for (int j = 0; j < BlockSize; j++)
                    for (int i = 0; i < BlockSize; i++)
                        if (maskIdx[r + j, c + i] != 0) maskCount++;

                double maskMean = (double)maskCount / (BlockSize * BlockSize);
                if (maskMean < roiThreshold) continue;

                int cy = r + BlockSize / 2;
                int cx = c + BlockSize / 2;
                double denom = gxxIdx[cy, cx] - gyyIdx[cy, cx];
                double numer = 2.0 * gxyIdx[cy, cx];
                double theta = (Math.PI + Math.Atan2(numer, denom)) / 2.0;

                double rotAngleDeg = theta * 180.0 / Math.PI + 90.0;
                using var blockMat = new Mat(BlockSize, BlockSize, MatType.CV_8UC1);
                for (int j = 0; j < BlockSize; j++)
                    for (int i = 0; i < BlockSize; i++)
                        blockMat.Set(j, i, grayIdx[r + j, c + i]);

                using var rotMatrix = Cv2.GetRotationMatrix2D(
                    new Point2f(BlockSize / 2f, BlockSize / 2f), rotAngleDeg, 1.0);
                using var rotatedMat = new Mat();
                Cv2.WarpAffine(blockMat, rotatedMat, rotMatrix, new Size(BlockSize, BlockSize));

                var rotIdx = rotatedMat.GetGenericIndexer<byte>();
                Array.Clear(projection, 0, BlockSize);
                for (int j = 0; j < BlockSize; j++)
                    for (int i = 0; i < BlockSize; i++)
                        projection[i] += rotIdx[j, i];

                var mag = Dft1D.Magnitude(projection);

                int peakIndex = 1;
                double peakVal = mag[1];
                int half = BlockSize / 2;
                for (int k = 2; k < half; k++)
                {
                    if (mag[k] > peakVal)
                    {
                        peakVal = mag[k];
                        peakIndex = k;
                    }
                }

                double freq = (double)peakIndex / BlockSize;
                double wavelength = freq > 0 ? 1.0 / freq : 0.0;
                if (wavelength > MinWavelength && wavelength < MaxWavelength)
                    frequencies.Add(freq);
            }
        }

        if (frequencies.Count == 0) return 0.0;

        double mean = frequencies.Average();
        double sumSq = 0.0;
        foreach (double f in frequencies) sumSq += (f - mean) * (f - mean);
        double std = Math.Sqrt(sumSq / frequencies.Count);

        return Math.Exp(-K * std);
    }
}
