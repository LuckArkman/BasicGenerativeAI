using System.Text.Json;
using System.Text.RegularExpressions;

namespace BasicGenerativeAI.Core;

public class BPETokenizer
{
    private Dictionary<string, int> _vocab; // token -> id
    private Dictionary<int, string> _inverseVocab; // id -> token
    private List<(string, string)> _mergesList; // Lista ordenada de mesclagens
    private Dictionary<(string, string), int> _mergeRanks; // (p1, p2) -> rank (prioridade)
    private Regex _pretokenizeRegex; // Regex de pré-tokenização do GPT-2

    // Cache para palavras já tokenizadas com BPE
    private Dictionary<string, List<string>> _bpeCache = new Dictionary<string, List<string>>();

    public BPETokenizer(string vocabPath, string mergesPath)
    {
        LoadVocab(vocabPath);
        LoadMerges(mergesPath);

        // Regex de pré-tokenização do GPT-2 (complexa!)
        // Esta é uma simplificação, a original é mais elaborada.
        // A regex original do GPT-2 é:
        // 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
        // Adaptar isso para .NET Regex pode ser necessário.
        // Para este exemplo, usaremos uma mais simples.
        _pretokenizeRegex = new Regex(@"'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]+|[^\s\p{L}\p{N}]+|\s", RegexOptions.Compiled);
    }

    private void LoadVocab(string vocabPath)
    {
        _vocab = JsonSerializer.Deserialize<Dictionary<string, int>>(File.ReadAllText(vocabPath));
        _inverseVocab = _vocab.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
    }

    private void LoadMerges(string mergesPath)
    {
        _mergesList = new List<(string, string)>();
        _mergeRanks = new Dictionary<(string, string), int>();
        var lines = File.ReadAllLines(mergesPath);
        // Pula a primeira linha (geralmente um comentário de versão)
        for (int i = 1; i < lines.Length; i++)
        {
            var parts = lines[i].Split(' ');
            if (parts.Length == 2)
            {
                var pair = (parts[0], parts[1]);
                _mergesList.Add(pair);
                _mergeRanks[pair] = i; // Rank baseado na ordem do arquivo
            }
        }
    }
    
    // Normalização de bytes (crucial para GPT-2, mas complexo de reimplementar 100%)
    // O GPT-2 mapeia bytes para caracteres Unicode específicos para evitar problemas com
    // caracteres de controle e garantir que cada byte tenha uma representação única.
    // Esta é uma parte difícil de acertar.
    private string ByteLevelNormalize(string text)
    {
        // Esta é uma GRANDE simplificação. A implementação real é mais envolvida
        // com um mapeamento byte -> char Unicode específico.
        // Por agora, vamos apenas retornar o texto. Na prática, isso quebraria a compatibilidade.
        return text; 
    }


    private List<string> BytePairEncode(string token)
    {
        if (_bpeCache.TryGetValue(token, out var cachedResult))
        {
            return cachedResult;
        }

        // No GPT-2, a tokenização opera em uma representação de bytes.
        // A palavra 'token' aqui já seria o resultado da pré-tokenização e normalização.
        // Para este exemplo, vamos assumir que 'token' é uma string de caracteres.
        List<string> wordChars = token.Select(c => c.ToString()).ToList();

        // O loop de mesclagem (como no pseudocódigo Python acima) iria aqui
        // Usando _mergeRanks para encontrar o par de menor rank (maior prioridade)
        // e mesclando-o iterativamente.

        // --- Início da Lógica de Mesclagem Simplificada ---
        while (true)
        {
            (string, string)? bestPairToMerge = null;
            int minRank = int.MaxValue;

            for (int i = 0; i < wordChars.Count - 1; i++)
            {
                var currentPair = (wordChars[i], wordChars[i + 1]);
                if (_mergeRanks.TryGetValue(currentPair, out int rank))
                {
                    if (rank < minRank)
                    {
                        minRank = rank;
                        bestPairToMerge = currentPair;
                    }
                }
            }

            if (!bestPairToMerge.HasValue) // Nenhuma mesclagem encontrada
            {
                break;
            }

            // Realiza a mesclagem
            var newWordChars = new List<string>();
            string first = bestPairToMerge.Value.Item1;
            string second = bestPairToMerge.Value.Item2;
            
            int j = 0;
            while(j < wordChars.Count)
            {
                if (j < wordChars.Count - 1 && wordChars[j] == first && wordChars[j+1] == second)
                {
                    newWordChars.Add(first + second);
                    j += 2;
                }
                else
                {
                    newWordChars.Add(wordChars[j]);
                    j += 1;
                }
            }
            wordChars = newWordChars;
        }
        // --- Fim da Lógica de Mesclagem Simplificada ---

        _bpeCache[token] = wordChars;
        return wordChars;
    }

    public List<int> Encode(string text)
    {
        string normalizedText = ByteLevelNormalize(text); // Passo crucial que simplificamos muito
        var ids = new List<int>();

        MatchCollection matches = _pretokenizeRegex.Matches(normalizedText);
        foreach (Match match in matches)
        {
            string preToken = match.Value;
            List<string> bpeTokens = BytePairEncode(preToken);
            foreach (string bpeToken in bpeTokens)
            {
                if (_vocab.TryGetValue(bpeToken, out int id))
                {
                    ids.Add(id);
                }
                else
                {
                    // Lidar com tokens não encontrados (em GPT-2, isso é raro devido ao nível de byte)
                    // Poderia quebrar em caracteres e tentar mapeá-los, ou usar um UNK se definido.
                    // A implementação real do GPT-2 mapeia para bytes individuais se um token maior não for encontrado.
                    Console.WriteLine($"Aviso: Token BPE '{bpeToken}' não encontrado no vocabulário.");
                    // Por simplicidade, não adicionaremos nada aqui, mas uma implementação real precisaria de um fallback.
                }
            }
        }
        return ids;
    }

    public string Decode(List<int> ids)
    {
        // A decodificação precisa reverter a normalização de bytes, o que é complexo.
        // Esta é uma simplificação extrema.
        return string.Concat(ids.Select(id => _inverseVocab.TryGetValue(id, out var token) ? token : ""));
    }
}