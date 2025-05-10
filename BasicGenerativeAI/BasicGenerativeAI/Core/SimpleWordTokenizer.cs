using System.Text.RegularExpressions;

namespace BasicGenerativeAI.Core;

public class SimpleWordTokenizer
{
    public Dictionary<string, int> WordToId { get; private set; }
    public Dictionary<int, string> IdToWord { get; private set; }
    public int VocabularySize => WordToId.Count;

    // Tokens especiais
    public const string UNK_TOKEN = "<UNK>"; // Token para palavras desconhecidas
    public const string PAD_TOKEN = "<PAD>"; // Token para preenchimento
    public int UnkTokenId { get; private set; }
    public int PadTokenId { get; private set; }

    private int _nextId = 0;

    public SimpleWordTokenizer()
    {
        WordToId = new Dictionary<string, int>();
        IdToWord = new Dictionary<int, string>();
        
        // Adiciona tokens especiais primeiro
        AddToken(PAD_TOKEN);
        PadTokenId = WordToId[PAD_TOKEN];

        AddToken(UNK_TOKEN);
        UnkTokenId = WordToId[UNK_TOKEN];
    }

    private void AddToken(string token)
    {
        if (!WordToId.ContainsKey(token))
        {
            WordToId[token] = _nextId;
            IdToWord[_nextId] = token;
            _nextId++;
        }
    }

    // "Treina" o vocabulário a partir de um texto
    public void BuildVocab(string text)
    {
        var tokens = TokenizeTextInternal(text);
        foreach (var token in tokens)
        {
            AddToken(token);
        }
        Console.WriteLine($"Vocabulário construído com {VocabularySize} tokens.");
    }

    // Tokeniza uma string de entrada
    private List<string> TokenizeTextInternal(string text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return new List<string>();
        }

        // Normalização básica: minúsculas
        text = text.ToLowerInvariant();

        // Divide por espaços e mantém pontuações como tokens separados
        // Regex para separar palavras e pontuações comuns.
        // \w+ pega palavras (letras, números, _)
        // [^\w\s] pega qualquer coisa que não seja palavra ou espaço (pontuações)
        var matches = Regex.Matches(text, @"\w+|[^\w\s]");
        
        return matches.Cast<Match>().Select(m => m.Value).ToList();
    }

    // Codifica texto para uma lista de IDs
    public List<int> Encode(string text)
    {
        var tokens = TokenizeTextInternal(text);
        var ids = new List<int>();
        foreach (var token in tokens)
        {
            if (WordToId.TryGetValue(token, out int id))
            {
                ids.Add(id);
            }
            else
            {
                ids.Add(UnkTokenId); // Palavra desconhecida
            }
        }
        return ids;
    }

    // Decodifica uma lista de IDs para texto
    public string Decode(List<int> ids)
    {
        var tokens = new List<string>();
        foreach (var id in ids)
        {
            if (IdToWord.TryGetValue(id, out string token))
            {
                // Opcional: não adicionar PAD tokens na decodificação
                if (id != PadTokenId) 
                {
                    tokens.Add(token);
                }
            }
            else
            {
                // Se um ID não for reconhecido (improvável se Encode foi usado),
                // poderia adicionar um placeholder ou ignorar.
                // Vamos adicionar UNK para consistência, embora não deveria acontecer.
                tokens.Add(UNK_TOKEN);
            }
        }
        // Junta os tokens. Para um tokenizador simples, um espaço é suficiente.
        // Tokenizadores mais avançados teriam lógica para reconstruir espaços corretamente.
        return string.Join(" ", tokens)
            .Replace(" ,", ",")
            .Replace(" .", ".")
            .Replace(" ?", "?")
            .Replace(" !", "!"); // Pós-processamento simples para pontuação
    }
}