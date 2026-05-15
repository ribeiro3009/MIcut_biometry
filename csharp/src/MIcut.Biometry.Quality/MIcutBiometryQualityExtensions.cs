using MIcut.Biometry.Quality.Extractors;

namespace MIcut.Biometry.Quality;

/// <summary>
/// Helpers de registro DI. Usar: <c>services.AddMIcutBiometryQualityExtractors();</c>.
/// O método NÃO depende de Microsoft.Extensions.DependencyInjection — recebe um delegate
/// genérico para evitar acoplamento adicional. Ver INTEGRACAO_NO_COLETOR.md para uso direto.
/// </summary>
public static class MIcutBiometryQualityRegistry
{
    public static IReadOnlyList<IMIcutQualityExtractor> CreateAll() => new IMIcutQualityExtractor[]
    {
        new MIcutMinutiaeCountExtractor(),
        new MIcutSingularitiesCountExtractor(),
        new MIcutClusterCountExtractor(),
        new MIcutSolidityExtractor(),
        new MIcutCoverageExtractor(),
        new MIcutSharpnessExtractor(),
        new MIcutOrientationStdExtractor(),
        new MIcutContrastExtractor(),
        new MIcutRidgeConsistencyExtractor(),
    };
}
