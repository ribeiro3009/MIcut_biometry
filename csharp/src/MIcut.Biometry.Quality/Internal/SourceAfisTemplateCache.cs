using System.Security.Cryptography;
using SourceAFIS;

namespace MIcut.Biometry.Quality.Internal;

internal sealed class SourceAfisTemplateCache
{
    public static SourceAfisTemplateCache Instance { get; } = new();

    private const int MaxEntries = 32;
    private readonly object _gate = new();
    private readonly Dictionary<string, TemplateData?> _cache = new(MaxEntries);

    public TemplateData? GetOrCreate(byte[] bmpBytes)
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

    private static TemplateData? TryBuild(byte[] bmpBytes)
    {
        try
        {
            var image = new FingerprintImage(bmpBytes, new FingerprintImageOptions { Dpi = 500 });
            var template = new FingerprintTemplate(image);
            return TemplateData.FromCbor(template.ToByteArray());
        }
        catch
        {
            return null;
        }
    }
}
