namespace MIcut.Biometry.Quality.Internal;

internal static class Dft1D
{
    public static double[] Magnitude(ReadOnlySpan<double> signal)
    {
        int n = signal.Length;
        var mag = new double[n];
        for (int k = 0; k < n; k++)
        {
            double re = 0.0;
            double im = 0.0;
            double twoPiKOverN = -2.0 * Math.PI * k / n;
            for (int t = 0; t < n; t++)
            {
                double angle = twoPiKOverN * t;
                re += signal[t] * Math.Cos(angle);
                im += signal[t] * Math.Sin(angle);
            }
            mag[k] = Math.Sqrt(re * re + im * im);
        }
        return mag;
    }
}
