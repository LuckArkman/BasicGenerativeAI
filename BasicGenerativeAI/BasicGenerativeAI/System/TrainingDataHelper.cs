using BasicGenerativeAI.Services;
using TorchSharp;
using System.Collections.Generic;
using System.Linq;

namespace BasicGenerativeAI.System;

public static class TrainingDataHelper
{
    // Prepara os dados de treinamento a partir de um texto bruto
    // text: A string contendo todo o texto de treinamento (pode ser concatenado de várias fontes)
    // tokenizerService: O serviço de tokenizer para converter texto em IDs
    // maxSequenceLength: O comprimento máximo das sequências de input e target
    // batchSize: O número de sequências por batch
    public static List<(torch.Tensor inputBatch, torch.Tensor targetBatch)> PrepareBatches(
        string text,
        TokenizerService tokenizerService,
        int maxSequenceLength,
        int batchSize)
    {
        // 1. Tokenizar o texto completo
        Console.WriteLine("Tokenizando texto de treinamento...");
        List<long> allTokenIds = tokenizerService.Encode(text);
        Console.WriteLine($"Total de tokens: {allTokenIds.Count}");

        // Adicionar o token de fim de sequência (EOS) no final, se não estiver lá
        // Isso é importante para o modelo aprender onde terminar a geração
        if (!tokenizerService.IsEndOfSequenceToken(allTokenIds.Last()))
        {
            allTokenIds.Add((long)tokenizerService.VocabularySize -
                            1); // EOS token ID para GPT-2 base é 50256, que é vocab_size - 1
        }


        // 2. Criar sequências de input e target
        // Input: [t1, t2, ..., tN]
        // Target: [t2, t3, ..., tN, tN+1]
        // Para um texto tokenizado [T1, T2, T3, T4, T5, EOS], com max_seq_len = 3
        // Sequência 1:
        // Input: [T1, T2] (length 2) - padded to [T1, T2, PAD]
        // Target: [T2, T3] (length 2) - padded to [T2, T3, PAD]
        // Sequência 2:
        // Input: [T2, T3] (length 2) - padded to [T2, T3, PAD]
        // Target: [T3, T4] (length 2) - padded to [T3, T4, PAD]
        // ...
        // Sequência K:
        // Input: [T_k, ..., T_{k+max_seq_len-2}]
        // Target: [T_{k+1}, ..., T_{k+max_seq_len-1}]

        // Vamos simplificar criando pares de subsequências deslizantes de tamanho maxSequenceLength + 1
        // A sequência de input terá tamanho maxSequenceLength (tokens 0 a maxSequenceLength-1)
        // A sequência de target terá tamanho maxSequenceLength (tokens 1 a maxSequenceLength)

        var sequences = new List<(torch.Tensor input, torch.Tensor target)>();
        var padTokenId =
            (long)tokenizerService
                .VocabularySize; // Um ID fora do vocabulário para padding (ou usar um token PAD real se o tokenizer tiver)
        // NOTA: Usar vocab_size como ID de PAD é uma convenção comum.
        // GPT-2 não tem PAD nativamente, então podemos usar um ID não usado ou o EOS.
        // Usar EOS pode ensinar o modelo a parar após padding. Usar um ID inválido requer ignorá-lo na loss.
        // Vamos usar EOS para simplificar e garantir que ele seja processado pelo Embedding.
        padTokenId = (long)tokenizerService.VocabularySize - 1; // Usar EOS como PAD token ID

        for (int i = 0; i < allTokenIds.Count - maxSequenceLength; i++)
        {
            // Pega uma fatia de tokens de tamanho maxSequenceLength para o input
            var inputSlice = allTokenIds.Skip(i).Take(maxSequenceLength).ToList();

            // Pega a fatia correspondente para o target (um token à frente)
            var targetSlice = allTokenIds.Skip(i + 1).Take(maxSequenceLength).ToList();

            // Converte para tensores Long (Int64)
            var inputTensor = torch.tensor(inputSlice.ToArray(), dtype: torch.ScalarType.Int64);
            var targetTensor = torch.tensor(targetSlice.ToArray(), dtype: torch.ScalarType.Int64);

            sequences.Add((inputTensor, targetTensor));
        }

        Console.WriteLine($"Total de sequências de treinamento criadas: {sequences.Count}");

        // 3. Embaralhar as sequências
        // Para um treinamento eficaz, é importante embaralhar os dados a cada época.
        // Vamos fazer um embaralhamento simples aqui.
        var rng = new Random();
        sequences = sequences.OrderBy(x => rng.Next()).ToList();


        // 4. Criar batches
        var batches = new List<(torch.Tensor inputBatch, torch.Tensor targetBatch)>();
        for (int i = 0; i < sequences.Count; i += batchSize)
        {
            var batchSequences = sequences.Skip(i).Take(batchSize).ToList();

            // Empilha as sequências para formar um batch tensor (batch_size, max_seq_len)
            // torch.stack espera uma lista de tensores com o mesmo shape.
            var inputBatch = torch.stack(batchSequences.Select(s => s.input).ToArray());
            var targetBatch = torch.stack(batchSequences.Select(s => s.target).ToArray());

            batches.Add((inputBatch, targetBatch));
        }

        // Dispor os tensores individuais das sequências após empilhá-los em batches
        foreach (var seq in sequences)
        {
            seq.input.Dispose();
            seq.target.Dispose();
        }

        sequences.Clear(); // Liberar a lista de referências


        Console.WriteLine($"Total de batches criados: {batches.Count}");
        return batches;
    }

    // Exemplo de dados de treinamento
    public static string ExampleTrainingData = @"
O gato sentou no tapete.
O cachorro latiu para o carteiro.
O pássaro cantou na árvore.
Era uma vez, em uma terra distante, um castelo.
A chuva caiu forte durante a noite.
O sol nasceu pela manhã.
Aprendi a programar em C#.
TorchSharp é uma biblioteca para .NET.
Modelos de linguagem geram texto.
Eu gosto de aprender coisas novas.
Qual é a capital da França? Paris.
Qual a cor do céu? Azul.
O que você está fazendo? Estou conversando com você.
Como está o tempo hoje? Não sei, não tenho acesso.
Buscar por melhores restaurantes em São Paulo.
Buscar por notícias recentes sobre tecnologia.
Busca por previsão do tempo em Nova Iorque.
";
}