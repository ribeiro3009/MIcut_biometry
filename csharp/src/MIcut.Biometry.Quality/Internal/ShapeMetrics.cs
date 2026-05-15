using OpenCvSharp;

namespace MIcut.Biometry.Quality.Internal;

internal static class ShapeMetrics
{
    public readonly record struct Result(double Solidity, double Coverage);

    public static Result Compute(Mat mask)
    {
        if (mask.Empty()) return new Result(0.0, 0.0);

        Cv2.FindContours(mask, out var contours, out _,
            RetrievalModes.External, ContourApproximationModes.ApproxSimple);
        if (contours.Length == 0) return new Result(0.0, 0.0);

        Point[] largest = contours[0];
        double largestArea = Cv2.ContourArea(largest);
        for (int i = 1; i < contours.Length; i++)
        {
            double a = Cv2.ContourArea(contours[i]);
            if (a > largestArea)
            {
                largest = contours[i];
                largestArea = a;
            }
        }

        double eps = 0.01 * Cv2.ArcLength(largest, true);
        Point[] approx = Cv2.ApproxPolyDP(largest, eps, true);
        double area = Cv2.ContourArea(approx);

        Point[] hull = Cv2.ConvexHull(approx);
        double hullArea = Cv2.ContourArea(hull);
        double solidity = hullArea > 0 ? area / hullArea : 0.0;

        Rect bb = Cv2.BoundingRect(approx);
        double bboxArea = (double)bb.Width * bb.Height;
        double coverage = bboxArea > 0 ? area / bboxArea : 0.0;

        return new Result(solidity, coverage);
    }
}
