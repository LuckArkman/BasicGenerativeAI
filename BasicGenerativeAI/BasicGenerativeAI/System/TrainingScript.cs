using BasicGenerativeAI.Core;
using BasicGenerativeAI.Services;
using TorchSharp;
using static TorchSharp.torch; // Mantido para torch.tensor, etc.
using TorchSharp.Modules; // Mantido para torch.tensor, etc.
using static TorchSharp.torch.optim;
using System.Collections.Generic;
using System.Linq;
using System.IO;
// using TorchSharp.Modules; // Não é estritamente necessário se usar torch.nn.Module

namespace BasicGenerativeAI.System;

public class TrainingScript : IDisposable // Implementa IDisposable
{
    private readonly TokenizerService _tokenizerService;
    private readonly TorchSharpGenerativeModel _model;
    private readonly int _epochs;
    private readonly int _batchSize;
    private readonly int _maxSequenceLength;
    private readonly double _learningRate;
    private readonly string _modelSavePath;
    private readonly Device _device;

    // Tratado CS8618 tornando-os anuláveis.
    private Optimizer? _optimizer;
    private CrossEntropyLoss? _criterion; // Usar TorchSharp.Modules.CrossEntropyLoss

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

    public void RunTraining()
    {
        if (!File.Exists("training_text.txt"))
        {
            Console.WriteLine("Erro: Arquivo 'training_text.txt' não encontrado. Crie este arquivo com texto para treinamento.");
            // Você pode adicionar aqui a lógica para criar um arquivo de exemplo se ele não existir:
            // File.WriteAllText("training_text.txt", TrainingDataHelper.ExampleTrainingData);
            // Console.WriteLine("Arquivo 'training_text.txt' criado com dados de exemplo. Edite-o e execute novamente.");
            return;
        }

        string trainingText = File.ReadAllText("training_text.txt");
        if (string.IsNullOrWhiteSpace(trainingText))
        {
            Console.WriteLine("Erro: O arquivo 'training_text.txt' está vazio.");
            return;
        }

        var trainingBatches = TrainingDataHelper.PrepareBatches(
            trainingText,
            _tokenizerService,
            _maxSequenceLength,
            _batchSize
        ).ToArray(); // ToArray() para materializar a coleção

        Console.WriteLine($"Total de batches de treinamento: {trainingBatches.Length}");
        if (!trainingBatches.Any()) // Usar Any() para verificar se a coleção está vazia
        {
            Console.WriteLine("Erro: Nenhum batch de treinamento foi criado. Verifique os dados de treinamento e maxSequenceLength.");
            return;
        }

        // Pegar apenas parâmetros treináveis
        var parameters = _model.parameters().Where(p => p.requires_grad).ToList();
        if (!parameters.Any())
        {
            Console.WriteLine("Erro: O modelo não possui parâmetros treináveis.");
            return;
        }
        Console.WriteLine($"Número de tensores de parâmetros treináveis: {parameters.Count()}");
        
        _optimizer = Adam(parameters, lr: _learningRate); // Outros hiperparâmetros do Adam têm defaults razoáveis
        _criterion = torch.nn.CrossEntropyLoss(); // Pode adicionar ignore_index: _tokenizerService.PadTokenId se o PAD ID for consistente

        _model.train(); // Coloca o modelo em modo de treinamento

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            Console.WriteLine($"\nÉpoca {epoch + 1}/{_epochs}...");
            float epochTotalLoss = 0f;
            int batchesProcessedThisEpoch = 0;

            foreach (var (inputBatchOriginal, targetBatchOriginal) in trainingBatches)
            {
                // Mover batches para o device ANTES de usá-los
                using var inputBatch = inputBatchOriginal.to(_device);
                using var targetBatch = targetBatchOriginal.to(_device);

                batchesProcessedThisEpoch++;
                // Descomente para log detalhado de cada batch:
                // Console.WriteLine($"  Batch {batchesProcessedThisEpoch}/{trainingBatches.Length}: Input shape: [{string.Join(", ", inputBatch.shape)}], Target shape: [{string.Join(", ", targetBatch.shape)}]");

                // Opcional: Verificação de validade dos IDs (já presente, mas pode ser útil)
                // long maxId = inputBatch.max().item<long>();
                // long minId = inputBatch.min().item<long>();
                // if (maxId >= _tokenizerService.VocabularySize || minId < 0)
                // {
                //     Console.WriteLine($"  Erro no Batch: IDs [{minId}-{maxId}] fora do intervalo do vocabulário [0, {_tokenizerService.VocabularySize - 1}].");
                //     continue; 
                // }

                _optimizer!.zero_grad(); // Usar '!' pois temos certeza que _optimizer é inicializado

                // outputLogits não deve ser disposed pelo _model.forward()
                using var outputLogits = _model.forward(inputBatch); 
                // outputLogits shape: (batch_size, sequence_length, vocab_size)

                if (outputLogits is null || outputLogits.Handle == IntPtr.Zero) // Verificação de segurança
                {
                    Console.WriteLine("Erro: outputLogits inválido após _model.forward().");
                    continue;
                }
                // Console.WriteLine($"  Output logits shape: [{string.Join(", ", outputLogits.shape)}]");


                // Reshape para CrossEntropyLoss:
                // Logits: (N, C) = (batch_size * sequence_length, vocab_size)
                // Targets: (N) = (batch_size * sequence_length)
                using var reshapedLogits = outputLogits.view(-1, _tokenizerService.VocabularySize);
                using var reshapedTargets = targetBatch.view(-1);

                using var loss = _criterion!.forward(reshapedLogits, reshapedTargets); // Usar '!'

                loss.backward();
                _optimizer.step();

                epochTotalLoss += loss.item<float>(); // Usar .item<float>()

                if (batchesProcessedThisEpoch % 10 == 0 || batchesProcessedThisEpoch == trainingBatches.Length)
                {
                    Console.WriteLine($"    Batch {batchesProcessedThisEpoch}/{trainingBatches.Length}, Loss Atual: {loss.item<float>():F4}, Loss Média Época (até agora): {epochTotalLoss / batchesProcessedThisEpoch:F4}");
                }
            }

            double avgLossThisEpoch = (batchesProcessedThisEpoch > 0) ? (epochTotalLoss / batchesProcessedThisEpoch) : 0.0;
            Console.WriteLine($"  Época {epoch + 1}/{_epochs} concluída. Loss média da época: {avgLossThisEpoch:F4}");

            if ((epoch + 1) % 5 == 0 || (epoch + 1) == _epochs) // Salvar a cada 5 épocas ou na última
            {
                string checkpointPath = _modelSavePath.Replace(".pth", $"_epoch{epoch + 1}.pth");
                _model.Save(checkpointPath);
                Console.WriteLine($"  Modelo salvo em checkpoint: {checkpointPath}");
            }
        }

        Console.WriteLine("\nTreinamento concluído.");
        _model.Save(_modelSavePath);
        Console.WriteLine($"Modelo final salvo em: {_modelSavePath}");
    }

    // Remover o método privado PrepareBatches se não for usado, pois TrainingDataHelper.PrepareBatches é usado.

    private bool disposedValue;
    protected virtual void Dispose(bool disposing)
    {
        if (!disposedValue)
        {
            if (disposing)
            {
                _optimizer?.Dispose();
                _criterion?.Dispose();
            }
            disposedValue = true;
        }
    }

    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }
}