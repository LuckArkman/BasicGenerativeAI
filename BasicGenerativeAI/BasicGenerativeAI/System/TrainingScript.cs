using BasicGenerativeAI.Core;
using BasicGenerativeAI.Services;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim; // Para otimizadores
using System.Collections.Generic;
using System.Linq;
using System.IO;
using TorchSharp.Modules; // Para caminhos de arquivo

namespace BasicGenerativeAI.System;

public class TrainingScript
{
    private readonly TokenizerService _tokenizerService;
    private readonly TorchSharpGenerativeModel _model;
    private readonly int _epochs;
    private readonly int _batchSize;
    private readonly int _maxSequenceLength;
    private readonly double _learningRate;
    private readonly string _modelSavePath;
    private readonly Device _device;
    private Optimizer _optimizer;
    private CrossEntropyLoss _criterion;

    // Construtor
    public TrainingScript(TokenizerService tokenizerService, TorchSharpGenerativeModel model,
        int epochs, int batchSize, int maxSequenceLength, double learningRate,
        string modelSavePath, Device device)
    {
        _tokenizerService = tokenizerService ?? throw new ArgumentNullException(nameof(tokenizerService));
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _epochs = epochs;
        _batchSize = batchSize;
        _maxSequenceLength = maxSequenceLength;
        _learningRate = learningRate;
        _modelSavePath = modelSavePath ?? throw new ArgumentNullException(nameof(modelSavePath));
        _device = device ?? torch.CPU;
    }

    // Executa o processo de treinamento
    public void RunTraining()
    {
        if (!File.Exists("training_text.txt"))
        {
            Console.WriteLine("Erro: Arquivo 'training_text.txt' não encontrado.");
            return;
        }

        string trainingText = File.ReadAllText("training_text.txt");
        var trainingBatches = TrainingDataHelper.PrepareBatches(
            trainingText,
            _tokenizerService,
            _maxSequenceLength,
            _batchSize
        ).ToArray();

        Console.WriteLine($"Total de batches: {trainingBatches.Length}");
        if (trainingBatches.Length == 0)
        {
            Console.WriteLine("Erro: Nenhum batch criado para treinamento.");
            return;
        }

        var parameters = _model.parameters();
        Console.WriteLine($"Número de parâmetros: {parameters.Count()}");
        _optimizer = Adam(
            parameters,
            lr: _learningRate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false
        );
        _criterion = CrossEntropyLoss();

        float totalLoss = 0f;
        int batchProcessed = 0;

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            Console.WriteLine($"Época {epoch + 1}/{_epochs}...");
            totalLoss = 0f;
            batchProcessed = 0;

            foreach (var (inputBatch, targetBatch) in trainingBatches)
            {
                batchProcessed++;
                Console.WriteLine(
                    $"Batch {batchProcessed}: Input shape: [{string.Join(", ", inputBatch.shape)}], Target shape: [{string.Join(", ", targetBatch.shape)}]");

                if (inputBatch.Handle == IntPtr.Zero)
                {
                    Console.WriteLine("Erro: inputBatch inválido antes de passar para o modelo.");
                    continue;
                }

                long maxId = inputBatch.max().item<long>();
                long minId = inputBatch.min().item<long>();
                Console.WriteLine($"Input batch - Máximo ID: {maxId}, Mínimo ID: {minId}");
                if (maxId >= _tokenizerService.VocabularySize || minId < 0)
                {
                    Console.WriteLine($"Erro: IDs fora do intervalo [0, {_tokenizerService.VocabularySize - 1}]");
                    continue;
                }

                using var outputLogits = _model.forward(inputBatch);
                if (outputLogits is null || outputLogits.Handle == IntPtr.Zero)
                {
                    Console.WriteLine("Erro: outputLogits inválido após forward.");
                    continue;
                }

                Console.WriteLine($"Output logits shape: [{string.Join(", ", outputLogits.shape)}]");

                using var reshapedLogits = outputLogits.view(
                    outputLogits.size(0) * outputLogits.size(1),
                    outputLogits.size(2)
                );
                using var reshapedTargets = targetBatch.view(targetBatch.size(0) * targetBatch.size(1));

                using var loss = _criterion.forward(reshapedLogits, reshapedTargets);

                _optimizer.zero_grad();
                loss.backward();
                _optimizer.step();

                totalLoss += loss.ToSingle();

                if (batchProcessed % 10 == 0)
                {
                    Console.WriteLine(
                        $"Época {epoch}/{_epochs}, Batch {batchProcessed}/{trainingBatches.Length}, Loss: {totalLoss / batchProcessed:F4}");
                }
            }

            double avgLoss = totalLoss / batchProcessed;
            Console.WriteLine($"Época {epoch}/{_epochs} concluída. Loss média: {avgLoss:F4}");

            if (epoch % 5 == 0 || epoch == _epochs - 1)
            {
                _model.Save($"{_modelSavePath.Replace(".pth", "")}_epoch{epoch}.pth");
            }
        }

        Console.WriteLine("Treinamento concluído.");
        _model.Save(_modelSavePath);

        _criterion?.Dispose();
    }

    private IEnumerable<(torch.Tensor, torch.Tensor)> PrepareBatches(long[] tokens, int batchSize,
        int maxSequenceLength)
    {
        for (int i = 0; i < tokens.Length - maxSequenceLength; i += batchSize * maxSequenceLength)
        {
            int batchEnd = Math.Min(i + batchSize * maxSequenceLength, tokens.Length);
            var batchTokens = tokens.Skip(i).Take(batchEnd - i).ToArray();

            for (int j = 0; j < batchTokens.Length - maxSequenceLength; j += maxSequenceLength)
            {
                var sequence = batchTokens.Skip(j).Take(maxSequenceLength).ToArray();
                if (sequence.Length < maxSequenceLength) continue;

                var input = torch.tensor(sequence.Take(maxSequenceLength - 1).ToArray(), dtype: ScalarType.Int64);
                var target = torch.tensor(sequence.Skip(1).Take(maxSequenceLength - 1).ToArray(),
                    dtype: ScalarType.Int64);

                if (input.shape[0] == maxSequenceLength - 1 && target.shape[0] == maxSequenceLength - 1)
                {
                    yield return (input, target);
                }
            }
        }
    }
}