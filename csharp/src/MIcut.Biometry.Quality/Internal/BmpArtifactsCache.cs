using System.Runtime.InteropServices;
using System.Security.Cryptography;
using OpenCvSharp;

namespace MIcut.Biometry.Quality.Internal;

internal sealed class BmpArtifactsCache
{
    public static BmpArtifactsCache Instance { get; } = new();

    public sealed record Artifacts(byte[] GrayPixels, byte[] MaskPixels, int Width, int Height)
    {
        public Mat NewGrayMat()
        {
            var mat = new Mat(Height, Width, MatType.CV_8UC1);
            Marshal.Copy(GrayPixels, 0, mat.Data, GrayPixels.Length);
            return mat;
        }

        public Mat NewMaskMat()
        {
            var mat = new Mat(Height, Width, MatType.CV_8UC1);
            Marshal.Copy(MaskPixels, 0, mat.Data, MaskPixels.Length);
            return mat;
        }
    }

    private const int MaxEntries = 16;
    private readonly object _gate = new();
    private readonly Dictionary<string, Artifacts?> _cache = new(MaxEntries);

    public Artifacts? GetOrCreate(byte[] bmpBytes)
    {
        string key = Convert.ToHexString(SHA1.HashData(bmpBytes));
        lock (_gate)
        {
            if (_cache.TryGetValue(key, out var cached)) return cached;
            var built = TryBuild(bmpBytes);
            if (_cache.Count >= MaxEntries) _cache.Clear();
            _cache[key] = built;
            return built;
        }
    }

    private static Artifacts? TryBuild(byte[] bmpBytes)
    {
        Mat? gray = null;
        Mat? mask = null;
        try
        {
            gray = BmpDecoder.DecodeGrayscale(bmpBytes);
            if (gray is null) return null;

            mask = RoiMaskBuilder.Build(gray);

            int n = gray.Rows * gray.Cols;
            var grayPx = new byte[n];
            var maskPx = new byte[n];

            if (!gray.IsContinuous() || !mask.IsContinuous())
            {
                using var grayCont = gray.Clone();
                using var maskCont = mask.Clone();
                Marshal.Copy(grayCont.Data, grayPx, 0, n);
                Marshal.Copy(maskCont.Data, maskPx, 0, n);
            }
            else
            {
                Marshal.Copy(gray.Data, grayPx, 0, n);
                Marshal.Copy(mask.Data, maskPx, 0, n);
            }

            return new Artifacts(grayPx, maskPx, gray.Cols, gray.Rows);
        }
        catch
        {
            return null;
        }
        finally
        {
            gray?.Dispose();
            mask?.Dispose();
        }
    }
}
