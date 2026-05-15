using System.Reflection;

namespace MIcut.Biometry.Quality.Tests;

internal static class ResourceLoader
{
    public static byte[] Load(string resourceFileName)
    {
        var asm = typeof(ResourceLoader).Assembly;
        string name = asm.GetManifestResourceNames()
            .FirstOrDefault(n => n.EndsWith("." + resourceFileName, StringComparison.Ordinal))
            ?? throw new FileNotFoundException($"Embedded resource not found: {resourceFileName}");
        using var stream = asm.GetManifestResourceStream(name)
            ?? throw new InvalidOperationException($"Could not open resource {name}");
        using var ms = new MemoryStream();
        stream.CopyTo(ms);
        return ms.ToArray();
    }
}
