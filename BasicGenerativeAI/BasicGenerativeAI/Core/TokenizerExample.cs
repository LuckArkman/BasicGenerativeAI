namespace BasicGenerativeAI.Core;

public class TokenizerExample
{
    public static void Main_SimpleWordTokenizer(string[] args)
    {
        var tokenizer = new SimpleWordTokenizer();

        string trainingText = "Olá mundo! Este é um teste. Olá novamente, mundo do tokenizer.";
        tokenizer.BuildVocab(trainingText);

        Console.WriteLine("\nVocabulário:");
        foreach (var kvp in tokenizer.WordToId)
        {
            Console.WriteLine($"'{kvp.Key}': {kvp.Value}");
        }

        string inputText = "Olá, este é um novo mundo para testar.";
        Console.WriteLine($"\nInput: {inputText}");

        var encodedIds = tokenizer.Encode(inputText);
        Console.WriteLine($"Encoded IDs: [{string.Join(", ", encodedIds)}]");

        var decodedText = tokenizer.Decode(encodedIds);
        Console.WriteLine($"Decoded Text: '{decodedText}'");

        Console.WriteLine($"\nUNK ID: {tokenizer.UnkTokenId} ({SimpleWordTokenizer.UNK_TOKEN})");
        Console.WriteLine($"PAD ID: {tokenizer.PadTokenId} ({SimpleWordTokenizer.PAD_TOKEN})");
    }
}