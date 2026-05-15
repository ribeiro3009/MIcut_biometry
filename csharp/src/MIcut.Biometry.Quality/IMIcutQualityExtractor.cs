namespace MIcut.Biometry.Quality;

public interface IMIcutQualityExtractor
{
    int ExtrairQualidade(int posicaoGrid, byte[]? imagemBmpBytes, bool ehAmputado);
    int ScoreMaximo { get; }
    string Nome { get; }
}
