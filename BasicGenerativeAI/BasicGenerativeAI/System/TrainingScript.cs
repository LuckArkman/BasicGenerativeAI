using BasicGenerativeAI.Core;
using BasicGenerativeAI.Services;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim; // Para otimizadores
using System.Collections.Generic;
using System.Linq;
using System.IO; // Para caminhos de arquivo

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
            _device = device ?? Device.CPU; // Default para CPU se não especificado
        }

        // Executa o processo de treinamento
        public void RunTraining()
        {
            Console.WriteLine($"Iniciando treinamento em {_device.type.ToString().ToUpper()}...");

            // Carrega dados de treinamento (usando o helper com dados de exemplo)
            var trainingBatches = TrainingDataHelper.PrepareBatches(
                TrainingDataHelper.ExampleTrainingData,
                _tokenizerService,
                _maxSequenceLength,
                _batchSize
            );

            if (trainingBatches.Count == 0)
            {
                Console.WriteLine("Nenhum dado de treinamento gerado. Abortando.");
                return;
            }

            // Define a função de perda (Cross-Entropy Loss para classificação de token)
            // O ignore_index é útil se você tiver um token de padding que não deve contribuir para a loss
            // Para GPT-2 usando EOS como padding, talvez não queiramos ignorar, mas é uma opção.
            // Vamos ignorar o ID que usamos para padding no helper se ele for diferente de um ID válido.
            // Se usamos EOS (vocab_size - 1), não ignoramos. Se usarmos vocab_size, ignoramos.
            // Vamos assumir que o ID usado para padding *não* deve ser ignorado para simplificar com EOS.
            // Se você usar um ID específico (ex: vocab_size), use ignore_index.
             using var criterion = CrossEntropyLoss(reduction: Reduction.Mean);
             // Se estivéssemos ignorando um PAD_ID específico (e.g., vocab_size):
             // using var criterion = CrossEntropyLoss(ignore_index: _tokenizerService.VocabularySize, reduction: Reduction.Mean);


            // Define o otimizador (AdamW é uma escolha comum)
            using var optimizer = AdamW(_model.modelModule.parameters(), lr: _learningRate);

            // Move o modelo para o dispositivo (CPU ou GPU)
            _model.modelModule.to(_device);

            Console.WriteLine($"Treinamento por {_epochs} épocas...");

            // Loop de treinamento principal
            for (int epoch = 1; epoch <= _epochs; epoch++)
            {
                _model.modelModule.train(); // Coloca o modelo em modo de treinamento

                double totalLoss = 0;
                int batchCount = 0;

                foreach (var batch in trainingBatches)
                {
                    // Move os dados do batch para o dispositivo
                    using var inputBatch = batch.inputBatch.to(_device);
                    using var targetBatch = batch.targetBatch.to(_device);

                    optimizer.zero_grad(); // Zera os gradientes acumulados

                    // Forward pass: obter logits do modelo
                    // Input shape: (batch_size, seq_len)
                    // Output logits shape: (batch_size, seq_len, vocab_size)
                    using var outputLogits = _model.modelModule.forward(inputBatch);

                    // Calcular a loss
                    // CrossEntropyLoss espera logits com shape (N, C) e targets com shape (N)
                    // Onde N é o número total de elementos (batch_size * seq_len)
                    // C é o número de classes (vocab_size)

                    // Reshape logits: (batch_size * seq_len, vocab_size)
                    using var reshapedLogits = outputLogits.view(
                         outputLogits.size(0) * outputLogits.size(1),
                         outputLogits.size(2)
                     );

                    // Reshape targets: (batch_size * seq_len)
                    using var reshapedTargets = targetBatch.view(targetBatch.size(0) * targetBatch.size(1));

                    // Calcula a loss
                    using var loss = criterion.forward(reshapedLogits, reshapedTargets);

                    // Backward pass: calcular gradientes
                    loss.backward();

                    // Otimizador step: atualizar pesos
                    optimizer.step();

                    totalLoss += loss.ToSingle(); // Acumula a loss (converte para float)
                    batchCount++;

                    // Opcional: imprimir loss a cada X batches
                    if (batchCount % 10 == 0)
                    {
                         Console.WriteLine($"  Época {epoch}/{_epochs}, Batch {batchCount}/{trainingBatches.Count}, Loss: {totalLoss / batchCount:F4}");
                    }

                     // Dispor tensores temporários no final de cada batch
                     inputBatch.Dispose();
                     targetBatch.Dispose();
                     outputLogits.Dispose();
                     reshapedLogits.Dispose();
                     reshapedTargets.Dispose();
                     loss.Dispose();
                }

                double avgLoss = totalLoss / trainingBatches.Count;
                Console.WriteLine($"Época {epoch}/{_epochs} concluída. Loss média: {avgLoss:F4}");

                // Opcional: Salvar o modelo a cada época ou a cada X épocas
                 if (epoch % 5 == 0 || epoch == _epochs) // Salva a cada 5 épocas e na última
                 {
                     _model.Save($"{_modelSavePath.Replace(".pth", "")}_epoch{epoch}.pth");
                 }
            }

            Console.WriteLine("Treinamento concluído.");

            // Salvar o modelo final
            _model.Save(_modelSavePath);

             // Dispor batches
             // Embora os tensores dos batches individuais tenham sido dispostos,
             // a lista de tuplas ainda pode conter referências.
             // Como PrepareBatches já dispôs os tensores individuais,
             // apenas limpar a lista aqui é suficiente.
             trainingBatches.Clear();


            // Dispor otimizador e loss (criados com using)
        }
    }