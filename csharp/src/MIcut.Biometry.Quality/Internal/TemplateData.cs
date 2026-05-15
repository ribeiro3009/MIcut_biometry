using System.Formats.Cbor;

namespace MIcut.Biometry.Quality.Internal;

internal sealed record TemplateData(int[] PositionsX, int[] PositionsY, int SingularitiesCount)
{
    public int MinutiaeCount => PositionsX.Length;

    public static TemplateData? FromCbor(byte[] cbor)
    {
        try
        {
            var reader = new CborReader(cbor, CborConformanceMode.Lax, allowMultipleRootLevelValues: false);
            if (reader.PeekState() != CborReaderState.StartMap) return null;
            reader.ReadStartMap();

            int[]? posX = null;
            int[]? posY = null;
            int? singularities = null;

            while (reader.PeekState() != CborReaderState.EndMap)
            {
                if (reader.PeekState() != CborReaderState.TextString)
                {
                    reader.SkipValue();
                    reader.SkipValue();
                    continue;
                }
                string key = reader.ReadTextString();
                switch (key)
                {
                    case "positionsX": posX = ReadIntArray(reader); break;
                    case "positionsY": posY = ReadIntArray(reader); break;
                    case "singularities": singularities = CountArrayElements(reader); break;
                    default: reader.SkipValue(); break;
                }
            }
            reader.ReadEndMap();

            return new TemplateData(
                posX ?? Array.Empty<int>(),
                posY ?? Array.Empty<int>(),
                singularities ?? 0);
        }
        catch
        {
            return null;
        }
    }

    private static int[] ReadIntArray(CborReader reader)
    {
        if (reader.PeekState() != CborReaderState.StartArray) { reader.SkipValue(); return Array.Empty<int>(); }
        int? len = reader.ReadStartArray();
        var list = new List<int>(len ?? 16);
        while (reader.PeekState() != CborReaderState.EndArray)
        {
            var state = reader.PeekState();
            if (state == CborReaderState.UnsignedInteger || state == CborReaderState.NegativeInteger)
            {
                list.Add(reader.ReadInt32());
            }
            else
            {
                reader.SkipValue();
            }
        }
        reader.ReadEndArray();
        return list.ToArray();
    }

    private static int CountArrayElements(CborReader reader)
    {
        if (reader.PeekState() != CborReaderState.StartArray) { reader.SkipValue(); return 0; }
        reader.ReadStartArray();
        int count = 0;
        while (reader.PeekState() != CborReaderState.EndArray)
        {
            reader.SkipValue();
            count++;
        }
        reader.ReadEndArray();
        return count;
    }
}
