using OpenCvSharp;

namespace MIcut.Biometry.Quality.Internal;

internal static class RoiMaskBuilder
{
    public static Mat Build(Mat gray)
    {
        using var thresh = new Mat();
        Cv2.AdaptiveThreshold(gray, thresh, 255,
            AdaptiveThresholdTypes.GaussianC, ThresholdTypes.BinaryInv, 21, 9);

        using var verticalKernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(1, 25));
        using var horizontalKernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(25, 1));
        using var verticalLines = new Mat();
        using var horizontalLines = new Mat();
        Cv2.MorphologyEx(thresh, verticalLines, MorphTypes.Open, verticalKernel);
        Cv2.MorphologyEx(thresh, horizontalLines, MorphTypes.Open, horizontalKernel);

        using var allLines = new Mat();
        Cv2.BitwiseOr(verticalLines, horizontalLines, allLines);

        using var notLines = new Mat();
        Cv2.BitwiseNot(allLines, notLines);

        using var fingerprintsOnly = new Mat();
        Cv2.BitwiseAnd(thresh, notLines, fingerprintsOnly);

        using var kernelSmall = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(3, 3));
        using var cleaned = new Mat();
        Cv2.MorphologyEx(fingerprintsOnly, cleaned, MorphTypes.Open, kernelSmall);

        using var kernelDilate5 = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(5, 5));
        using var filtered = new Mat();
        Cv2.Dilate(cleaned, filtered, kernelDilate5, iterations: 1);

        // Post-processing per ml_segmentation.process_and_save_crop: CLOSE(7,7) + dilate(7,7)
        using var kernel7 = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(7, 7));
        using var closed = new Mat();
        Cv2.MorphologyEx(filtered, closed, MorphTypes.Close, kernel7);

        var result = new Mat();
        Cv2.Dilate(closed, result, kernel7, iterations: 1);
        return result;
    }
}
