using BasicGenerativeAI.Services;
using TorchSharp;
using System.Collections.Generic;
using System.Linq;

namespace BasicGenerativeAI.System;

public static class TrainingDataHelper
{
    public static List<(torch.Tensor inputBatch, torch.Tensor targetBatch)> PrepareBatches(
    string text,
    TokenizerService tokenizerService,
    int maxSequenceLength,
    int batchSize)
{
    Console.WriteLine("Tokenizando texto de treinamento...");
    Console.WriteLine($"Texto bruto (primeiros 500 caracteres): {text.Substring(0, Math.Min(500, text.Length))}");
    var tokenIds = tokenizerService.TokenizeText(text).ToList();
    Console.WriteLine($"Total de tokens: {tokenIds.Count}");
    Console.WriteLine($"Primeiros 10 tokens: [{string.Join(", ", tokenIds.Take(10))}]");

    if (!tokenizerService.IsEndOfSequenceToken(tokenIds.Last()))
    {
        tokenIds.Add((long)tokenizerService.EndOfSequenceTokenId);
    }

    var sequences = new List<(torch.Tensor input, torch.Tensor target)>();
    var padTokenId = (long)tokenizerService.PadTokenId;

    int stepSize = Math.Max(1, maxSequenceLength - 1);
    for (int i = 0; i < tokenIds.Count - 1; i += stepSize)
    {
        var inputSlice = tokenIds.Skip(i).Take(maxSequenceLength).ToList();
        var targetSlice = tokenIds.Skip(i + 1).Take(maxSequenceLength).ToList();

        while (inputSlice.Count < maxSequenceLength) inputSlice.Add(padTokenId);
        while (targetSlice.Count < maxSequenceLength) targetSlice.Add(padTokenId);

        var inputTensor = torch.tensor(inputSlice.ToArray(), dtype: torch.ScalarType.Int64);
        var targetTensor = torch.tensor(targetSlice.ToArray(), dtype: torch.ScalarType.Int64);

        sequences.Add((inputTensor, targetTensor));
    }

    if (sequences.Count == 0 && tokenIds.Count > 1)
    {
        var inputSlice = tokenIds.Take(maxSequenceLength).ToList();
        var targetSlice = tokenIds.Skip(1).Take(maxSequenceLength).ToList();
        while (inputSlice.Count < maxSequenceLength) inputSlice.Add(padTokenId);
        while (targetSlice.Count < maxSequenceLength) targetSlice.Add(padTokenId);
        var inputTensor = torch.tensor(inputSlice.ToArray(), dtype: torch.ScalarType.Int64);
        var targetTensor = torch.tensor(targetSlice.ToArray(), dtype: torch.ScalarType.Int64);
        sequences.Add((inputTensor, targetTensor));
    }

    Console.WriteLine($"Total de sequências de treinamento criadas: {sequences.Count}");

    var rng = new Random();
    sequences = sequences.OrderBy(x => rng.Next()).ToList();

    var batches = new List<(torch.Tensor inputBatch, torch.Tensor targetBatch)>();
    for (int i = 0; i < sequences.Count; i += batchSize)
    {
        var batchSequences = sequences.Skip(i).Take(batchSize).ToList();

        while (batchSequences.Count < batchSize)
        {
            var dummyInput = torch.full(new long[] { maxSequenceLength }, padTokenId, dtype: torch.ScalarType.Int64);
            var dummyTarget = torch.full(new long[] { maxSequenceLength }, padTokenId, dtype: torch.ScalarType.Int64);
            batchSequences.Add((dummyInput, dummyTarget));
        }

        var inputBatch = torch.stack(batchSequences.Select(s => s.input).ToArray());
        var targetBatch = torch.stack(batchSequences.Select(s => s.target).ToArray());

        batches.Add((inputBatch, targetBatch));
    }

    foreach (var seq in sequences)
    {
        seq.input.Dispose();
        seq.target.Dispose();
    }
    sequences.Clear();

    Console.WriteLine($"Total de batches criados: {batches.Count}");
    return batches;
}

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