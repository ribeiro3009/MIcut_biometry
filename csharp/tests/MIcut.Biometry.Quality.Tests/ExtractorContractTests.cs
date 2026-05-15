using MIcut.Biometry.Quality;
using Xunit;

namespace MIcut.Biometry.Quality.Tests;

public class ExtractorContractTests
{
    public static IEnumerable<object[]> AllExtractors()
    {
        foreach (var e in MIcutBiometryQualityRegistry.CreateAll())
            yield return new object[] { e };
    }

    [Theory]
    [MemberData(nameof(AllExtractors))]
    public void Amputado_RetornaZero(IMIcutQualityExtractor extractor)
    {
        byte[] bmp = ResourceLoader.Load("finger_sample.bmp");
        int score = extractor.ExtrairQualidade(posicaoGrid: 0, imagemBmpBytes: bmp, ehAmputado: true);
        Assert.Equal(0, score);
    }

    [Theory]
    [MemberData(nameof(AllExtractors))]
    public void BmpNulo_RetornaZero(IMIcutQualityExtractor extractor)
    {
        int score = extractor.ExtrairQualidade(posicaoGrid: 0, imagemBmpBytes: null, ehAmputado: false);
        Assert.Equal(0, score);
    }

    [Theory]
    [MemberData(nameof(AllExtractors))]
    public void BmpVazio_RetornaZero(IMIcutQualityExtractor extractor)
    {
        int score = extractor.ExtrairQualidade(posicaoGrid: 0, imagemBmpBytes: Array.Empty<byte>(), ehAmputado: false);
        Assert.Equal(0, score);
    }

    [Theory]
    [MemberData(nameof(AllExtractors))]
    public void BmpInvalido_RetornaZero(IMIcutQualityExtractor extractor)
    {
        byte[] garbage = new byte[] { 0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE };
        int score = extractor.ExtrairQualidade(posicaoGrid: 0, imagemBmpBytes: garbage, ehAmputado: false);
        Assert.Equal(0, score);
    }

    [Theory]
    [MemberData(nameof(AllExtractors))]
    public void NomeNaoVazio_E_ScoreMaximoPositivo(IMIcutQualityExtractor extractor)
    {
        Assert.False(string.IsNullOrWhiteSpace(extractor.Nome));
        Assert.True(extractor.ScoreMaximo > 0, $"{extractor.Nome} ScoreMaximo deve ser > 0");
    }

    [Theory]
    [MemberData(nameof(AllExtractors))]
    public void BmpReal_ScoreDentroDoLimite(IMIcutQualityExtractor extractor)
    {
        byte[] bmp = ResourceLoader.Load("finger_sample.bmp");
        int score = extractor.ExtrairQualidade(posicaoGrid: 0, imagemBmpBytes: bmp, ehAmputado: false);
        Assert.InRange(score, 0, extractor.ScoreMaximo);
    }
}
