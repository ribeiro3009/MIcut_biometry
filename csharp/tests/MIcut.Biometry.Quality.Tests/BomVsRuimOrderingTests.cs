using MIcut.Biometry.Quality;
using MIcut.Biometry.Quality.Extractors;
using Xunit;
using Xunit.Abstractions;

namespace MIcut.Biometry.Quality.Tests;

public class BomVsRuimOrderingTests
{
    private readonly ITestOutputHelper _output;
    private readonly byte[] _good;
    private readonly byte[] _bad;

    public BomVsRuimOrderingTests(ITestOutputHelper output)
    {
        _output = output;
        _good = ResourceLoader.Load("finger_sample.bmp");
        _bad = BadBmpBuilder.MakeBadFromGood(_good, blurKernel: 51);
    }

    private void AssertBomMaiorQueRuim(IMIcutQualityExtractor extractor)
    {
        int sGood = extractor.ExtrairQualidade(0, _good, false);
        int sBad = extractor.ExtrairQualidade(0, _bad, false);
        _output.WriteLine($"{extractor.Nome}: good={sGood}, bad={sBad}");
        Assert.True(sGood > sBad,
            $"{extractor.Nome}: esperava good > bad, recebi good={sGood}, bad={sBad}");
    }

    [Fact] public void Sharpness_BomMaiorQueRuim()         => AssertBomMaiorQueRuim(new MIcutSharpnessExtractor());
    [Fact] public void RidgeConsistency_BomMaiorQueRuim()  => AssertBomMaiorQueRuim(new MIcutRidgeConsistencyExtractor());
    [Fact] public void OrientationStd_BomMaiorQueRuim()    => AssertBomMaiorQueRuim(new MIcutOrientationStdExtractor());
    [Fact] public void Contrast_BomMaiorQueRuim()          => AssertBomMaiorQueRuim(new MIcutContrastExtractor());
    [Fact] public void MinutiaeCount_BomMaiorQueRuim()     => AssertBomMaiorQueRuim(new MIcutMinutiaeCountExtractor());

    [Fact]
    public void BmpReal_ScoresPlausiveis()
    {
        foreach (var e in MIcutBiometryQualityRegistry.CreateAll())
        {
            int s = e.ExtrairQualidade(0, _good, false);
            _output.WriteLine($"{e.Nome,-30} = {s,5} / {e.ScoreMaximo}");
            Assert.InRange(s, 0, e.ScoreMaximo);
        }
    }
}
