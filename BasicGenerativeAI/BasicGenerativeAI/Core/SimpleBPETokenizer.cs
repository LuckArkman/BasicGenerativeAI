using System.Text;
using System.Text.RegularExpressions;

namespace BasicGenerativeAI.Core;

public class SimpleBPETokenizer
{
    public Dictionary<string, int> Vocab { get; private set; }
    public Dictionary<int, string> InverseVocab { get; private set; }
    
    // Ordem das mesclagens (pares mais frequentes primeiro)
    // Em um BPE real, isso seria aprendido do corpus.
    // Ex: ("e", "s") -> "es", ("es", "t") -> "est"
    private List<(string, string)> _merges; 
    private Dictionary<string, int> _mergeRanks; // Para aplicar mesclagens na ordem correta

    public int VocabularySize => Vocab.Count;
    private int _nextId = 0;

    // Tokens especiais
    public const string UNK_TOKEN = "<UNK>";
    public int UnkTokenId { get; private set; }
    // Para BPE, o padding e outros tokens especiais seriam parte do vocabulário aprendido ou adicionados
    // explicitamente.

    public SimpleBPETokenizer()
    {
        Vocab = new Dictionary<string, int>();
        InverseVocab = new Dictionary<int, string>();
        _merges = new List<(string, string)>();
        _mergeRanks = new Dictionary<string, int>();

        // Inicializa com caracteres individuais (ASCII básico para exemplo)
        for (int i = 32; i < 127; i++) // Caracteres imprimíveis ASCII
        {
            AddToken(((char)i).ToString());
        }
        // Adiciona tokens especiais
        AddToken(UNK_TOKEN); // Adiciona após os caracteres base
        UnkTokenId = Vocab[UNK_TOKEN];
    }

    private void AddToken(string token)
    {
        if (!Vocab.ContainsKey(token))
        {
            Vocab[token] = _nextId;
            InverseVocab[_nextId] = token;
            _nextId++;
        }
    }

    // Em um BPE real, esta função "aprenderia" as mesclagens de um corpus
    public void Train(List<string> corpusWords, int numMerges)
    {
        // 1. Inicializar vocabulário com caracteres únicos
        // (Já feito no construtor para este exemplo simplificado)

        // 2. Pré-tokenizar palavras em caracteres (ou unidades base)
        var wordSplits = new List<List<string>>();
        foreach (var word in corpusWords)
        {
            // Adiciona um sufixo especial para marcar fim de palavra e diferenciar de subpalavras internas
            // GPT-2 usa "Ġ" (espaço) no início de palavras. Outros usam "</w>" no fim.
            // Vamos simplificar e apenas dividir em caracteres por enquanto.
            wordSplits.Add(word.Select(c => c.ToString()).ToList());
        }

        // 3. Iterativamente encontrar o par mais frequente e mesclá-lo
        for (int i = 0; i < numMerges; i++)
        {
            var pairCounts = GetPairCounts(wordSplits);
            if (!pairCounts.Any()) break;

            var bestPair = pairCounts.OrderByDescending(kvp => kvp.Value).First().Key;
            
            // Adiciona a nova mesclagem ao vocabulário e à lista de mesclagens
            string mergedToken = bestPair.Item1 + bestPair.Item2;
            AddToken(mergedToken);
            _merges.Add(bestPair);
            _mergeRanks[mergedToken] = _merges.Count; // Rank é a ordem da mesclagem

            // Atualiza as divisões de palavras com a nova mesclagem
            wordSplits = MergeInSplits(wordSplits, bestPair, mergedToken);
            Console.WriteLine($"Merge {i+1}/{numMerges}: '{bestPair.Item1}' + '{bestPair.Item2}' -> '{mergedToken}'");
        }
        Console.WriteLine($"Treinamento BPE concluído. Vocabulário com {VocabularySize} tokens.");
    }
    
    private Dictionary<(string, string), int> GetPairCounts(List<List<string>> wordSplits)
    {
        var counts = new Dictionary<(string, string), int>();
        foreach (var split in wordSplits)
        {
            for (int j = 0; j < split.Count - 1; j++)
            {
                var pair = (split[j], split[j + 1]);
                counts[pair] = counts.GetValueOrDefault(pair, 0) + 1;
            }
        }
        return counts;
    }

    private List<List<string>> MergeInSplits(List<List<string>> oldSplits, (string p1, string p2) pairToMerge, string mergedToken)
    {
        var newSplits = new List<List<string>>();
        foreach (var split in oldSplits)
        {
            var newSplit = new List<string>();
            int j = 0;
            while (j < split.Count)
            {
                if (j < split.Count - 1 && split[j] == pairToMerge.p1 && split[j + 1] == pairToMerge.p2)
                {
                    newSplit.Add(mergedToken);
                    j += 2;
                }
                else
                {
                    newSplit.Add(split[j]);
                    j += 1;
                }
            }
            newSplits.Add(newSplit);
        }
        return newSplits;
    }

    // Aplica as mesclagens BPE a uma única palavra
    private List<string> BPEEncodeWord(string word)
    {
        if (string.IsNullOrEmpty(word)) return new List<string>();

        // GPT-2 e outros tokenizadores BPE têm uma etapa de normalização aqui
        // e frequentemente lidam com espaços de forma especial (ex: prefixando palavras com 'Ġ').
        // Para simplificar, vamos apenas dividir em caracteres.
        List<string> tokens = word.Select(c => c.ToString()).ToList();

        while (true)
        {
            var minRank = int.MaxValue;
            (int, int) bestPairIndices = (-1, -1); // (índice do primeiro, índice do segundo)
            string bestMergedToken = null;

            for (int i = 0; i < tokens.Count - 1; i++)
            {
                string pairStr = tokens[i] + tokens[i+1];
                if (_mergeRanks.TryGetValue(pairStr, out int rank))
                {
                    if (rank < minRank)
                    {
                        minRank = rank;
                        bestPairIndices = (i, i + 1);
                        bestMergedToken = pairStr;
                    }
                }
            }

            if (bestPairIndices.Item1 == -1) // Nenhuma mesclagem aplicável encontrada
            {
                break;
            }

            // Realiza a melhor mesclagem encontrada
            var newTokens = new List<string>();
            newTokens.AddRange(tokens.Take(bestPairIndices.Item1)); // Parte antes do par
            newTokens.Add(bestMergedToken);                         // O par mesclado
            newTokens.AddRange(tokens.Skip(bestPairIndices.Item2 + 1)); // Parte depois do par
            tokens = newTokens;
        }
        return tokens;
    }

    public List<int> Encode(string text)
    {
        var ids = new List<int>();
        // Uma tokenização BPE real teria uma etapa de "pré-tokenização" mais robusta
        // que lida com espaços, pontuação, etc., antes de aplicar BPE a cada "palavra".
        // Regex para dividir em palavras e pontuações (simplificado)
        var wordMatches = Regex.Matches(text, @"\w+|[^\w\s]+|\s+"); 
        
        foreach (Match match in wordMatches)
        {
            string segment = match.Value;
            if (string.IsNullOrWhiteSpace(segment)) // Lidar com espaços
            {
                // Tokenizadores BPE reais geralmente codificam espaços explicitamente
                // ou os anexam a palavras (ex: "Ġhello").
                // Para este exemplo simples, vamos tentar codificar o espaço se ele estiver no vocabulário.
                if (Vocab.TryGetValue(segment, out int spaceId)) {
                    ids.Add(spaceId);
                } // Senão, podemos ignorá-lo ou adicionar UNK
                continue;
            }

            var subwordTokens = BPEEncodeWord(segment);
            foreach (var subToken in subwordTokens)
            {
                if (Vocab.TryGetValue(subToken, out int id))
                {
                    ids.Add(id);
                }
                else
                {
                    // Se um subtoken não estiver no vocabulário (improvável se os caracteres base estiverem), use UNK
                    // Ou, em um BPE mais robusto, quebre o subtoken desconhecido em caracteres conhecidos.
                    foreach (char c in subToken) // Fallback para caracteres
                    {
                        if (Vocab.TryGetValue(c.ToString(), out int charId))
                        {
                            ids.Add(charId);
                        }
                        else
                        {
                             ids.Add(UnkTokenId);
                        }
                    }
                }
            }
        }
        return ids;
    }

    public string Decode(List<int> ids)
    {
        var sb = new StringBuilder();
        foreach (var id in ids)
        {
            if (InverseVocab.TryGetValue(id, out string token))
            {
                sb.Append(token);
            }
            else
            {
                sb.Append(UNK_TOKEN); 
            }
        }
        // A decodificação BPE pode precisar de pós-processamento para remover artefatos
        // como o 'Ġ' do GPT-2 ou juntar subpalavras corretamente.
        // Esta decodificação simples apenas concatena.
        return sb.ToString();
    }
}


// --- Exemplo de Uso BPE ---
public class TokenizerExampleBPE
{
    public static void Main(string[] args) // Mude para Main para testar este
    {
        var bpeTokenizer = new SimpleBPETokenizer();

        // Palavras do corpus para treinar (muito pequeno para um BPE real)
        var corpus = new List<string> { "lowest", "newer", "wider", "low", "new" };
        bpeTokenizer.Train(corpus, numMerges: 20); // Tentar aprender algumas mesclagens

        Console.WriteLine("\nVocabulário BPE final:");
        // Ordena para melhor visualização
        foreach (var kvp in bpeTokenizer.Vocab.OrderBy(kvp => kvp.Value)) 
        {
            Console.WriteLine($"'{kvp.Key}': {kvp.Value}");
        }

        string inputText = "this is the lowest newer item.";
        Console.WriteLine($"\nInput: {inputText}");

        var encodedBpeIds = bpeTokenizer.Encode(inputText);
        Console.WriteLine($"Encoded BPE IDs: [{string.Join(", ", encodedBpeIds)}]");
        Console.Write("Encoded BPE Tokens: ");
        foreach(var id in encodedBpeIds) { Console.Write($"'{bpeTokenizer.InverseVocab[id]}', "); }
        Console.WriteLine();

        var decodedBpeText = bpeTokenizer.Decode(encodedBpeIds);
        Console.WriteLine($"Decoded BPE Text: '{decodedBpeText}'");
    }
}