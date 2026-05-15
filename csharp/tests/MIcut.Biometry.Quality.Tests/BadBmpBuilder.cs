using OpenCvSharp;

namespace MIcut.Biometry.Quality.Tests;

internal static class BadBmpBuilder
{
    /// <summary>
    /// Builds a "bad" version of the given BMP by applying a heavy Gaussian blur.
    /// Used as the "ruim" side of the bom-vs-ruim ordering tests for mask-dependent extractors.
    /// </summary>
    public static byte[] MakeBadFromGood(byte[] goodBmp, int blurKernel = 51)
    {
        using var src = Cv2.ImDecode(goodBmp, ImreadModes.Grayscale);
        if (src.Empty()) throw new InvalidOperationException("Could not decode good BMP for blurring.");

        using var blurred = new Mat();
        Cv2.GaussianBlur(src, blurred, new Size(blurKernel, blurKernel), 0);

        return blurred.ToBytes(".bmp");
    }
}
