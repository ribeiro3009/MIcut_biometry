using OpenCvSharp;

namespace MIcut.Biometry.Quality.Internal;

internal static class BmpDecoder
{
    public static Mat? DecodeGrayscale(byte[] bmpBytes)
    {
        if (bmpBytes is null || bmpBytes.Length == 0) return null;
        Mat? mat = null;
        try
        {
            mat = Cv2.ImDecode(bmpBytes, ImreadModes.Grayscale);
            if (mat.Empty())
            {
                mat.Dispose();
                return null;
            }
            return mat;
        }
        catch
        {
            mat?.Dispose();
            return null;
        }
    }
}
