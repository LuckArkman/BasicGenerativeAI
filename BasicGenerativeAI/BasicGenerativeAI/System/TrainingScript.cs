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
        private readonly Device _device; // CPU ou GPU
        private OptimizerHelper _optimizer; // O Optimizer não é IDisposable
        private Loss<Tensor, Tensor, Tensor> _criterion; // Loss é IDisposable

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
            _modelSavePath = modelSavePath;
            _device = device ?? torch.CPU; // Default para CPU se não especificado
        }

        // Executa o processo de treinamento
        public void RunTraining()
    {
        // Usar o tokenizerService injetado via construtor
        var trainingText = File.ReadAllText("training_text.txt"); // Certifique-se de que o arquivo existe
        var trainingBatches = TrainingDataHelper.PrepareBatches(trainingText, _tokenizerService, maxSequenceLength: _maxSequenceLength, batchSize: _batchSize);

        // Inicializar otimizador e critério
        _optimizer = Adam(_model.parameters(), lr: _learningRate);
        _criterion = CrossEntropyLoss();

        Console.WriteLine($"Total de batches: {trainingBatches.Count}");
        if (trainingBatches.Count == 0)
        {
            Console.WriteLine("Erro: Nenhum batch criado para treinamento.");
            return;
        }

        float totalLoss = 0f;
        int batchCount = 0;

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            Console.WriteLine($"Época {epoch + 1}/{_epochs}...");
            batchCount = 0;
            totalLoss = 0f;

            foreach (var (inputBatch, targetBatch) in trainingBatches)
            {
                batchCount++;
                Console.WriteLine($"Batch {batchCount}: Input shape: [{string.Join(", ", inputBatch.shape)}], Target shape: [{string.Join(", ", targetBatch.shape)}]");

                // Validar o inputBatch
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

                // Forward pass único
                using var outputLogits = _model._modelModule.forward(inputBatch);
                // Verificar validade do Tensor de forma explícita
                bool isOutputNull = outputLogits is null;
                bool isHandleInvalid = !isOutputNull && outputLogits.Handle == IntPtr.Zero;
                bool isOutputInvalid = isOutputNull || isHandleInvalid;
                if (isOutputInvalid)
                {
                    Console.WriteLine("Erro: outputLogits inválido após forward.");
                    Console.WriteLine($"Input batch max ID: {maxId}, min ID: {minId}");
                    Console.WriteLine($"Input batch values (primeiros 10): [{string.Join(", ", inputBatch.flatten().to_type(ScalarType.Int64).cpu().data<long>().Take(10))}]");
                    continue;
                }

                Console.WriteLine($"Output logits shape: [{string.Join(", ", outputLogits.shape)}]");

                // Reshape logits e targets
                using var reshapedLogits = outputLogits.view(
                    outputLogits.size(0) * outputLogits.size(1),
                    outputLogits.size(2)
                );
                using var reshapedTargets = targetBatch.view(targetBatch.size(0) * targetBatch.size(1));

                // Calcular a perda
                using var loss = _criterion.forward(reshapedLogits, reshapedTargets);

                // Backward pass
                _optimizer.zero_grad();
                loss.backward();
                _optimizer.step();

                totalLoss += loss.ToSingle();
                batchCount++;

                if (batchCount % 10 == 0)
                {
                    Console.WriteLine($"  Época {epoch}/{_epochs}, Batch {batchCount}/{trainingBatches.Count}, Loss: {totalLoss / batchCount:F4}");
                }
            }

            double avgLoss = totalLoss / batchCount;
            Console.WriteLine($"Época {epoch}/{_epochs} concluída. Loss média: {avgLoss:F4}");

            // Salvar o modelo a cada 5 épocas e na última
            if (epoch % 5 == 0 || epoch == _epochs - 1)
            {
                _model.Save($"{_modelSavePath.Replace(".pth", "")}_epoch{epoch}.pth");
            }
        }

        Console.WriteLine("Treinamento concluído.");
        _model.Save(_modelSavePath);

        // Limpar referências (os tensores já foram dispostos com using)
        trainingBatches.Clear();
    }
    }