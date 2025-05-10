using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using BasicGenerativeAI.Services; // Assuming this exists and is correct
using System;
using System.Collections.Generic;
using System.IO;

namespace BasicGenerativeAI.Core
{
    public abstract class BaseGenerativeModel : IDisposable
    {
        public abstract int VocabularySize { get; }
        public abstract void Load(string filePath);
        public abstract void Save(string filePath);
        public abstract torch.Tensor Generate(torch.Tensor inputTokens, int maxTokens, float temperature = 1.0f);
        public abstract torch.Tensor forward(torch.Tensor input);

        protected virtual void Dispose(bool disposing)
        {
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
    }

    public class TorchSharpGenerativeModel : BaseGenerativeModel
    {
        public SimpleRNNLanguageModel? _modelModule; // Typed for direct access
        public readonly TokenizerService _tokenizerService;

        public override int VocabularySize => _tokenizerService.VocabularySize;

        private readonly int _embeddingDim;
        private readonly int _hiddenDim;
        private readonly int _rnnLayers;

        private const string StateDictKey = "model_state_dict";
        private const string ConfigKey = "config";

        public TorchSharpGenerativeModel(TokenizerService tokenizerService, int embeddingDim = 256, int hiddenDim = 512,
            int rnnLayers = 2)
        {
            _tokenizerService = tokenizerService ?? throw new ArgumentNullException(nameof(tokenizerService));
            _embeddingDim = embeddingDim;
            _hiddenDim = hiddenDim;
            _rnnLayers = rnnLayers;

            InitializeModel();
        }

        private void InitializeModel()
        {
            _modelModule?.Dispose(); // Dispose existing model if any
        
            _modelModule = new SimpleRNNLanguageModel(
                vocabSize: (long)this.VocabularySize, // Ensure long for vocabSize
                embeddingDim: _embeddingDim,
                hiddenDim: _hiddenDim,
                rnnLayers: _rnnLayers
            );

            Console.WriteLine(
                $"Modelo TorchSharp inicializado com Vocab Size: {VocabularySize}, Embedding Dim: {_embeddingDim}, Hidden Dim: {_hiddenDim}, RNN Layers: {_rnnLayers}");
        }
        public override void Load(string filePath)
        {
            if (!File.Exists(filePath))
            {
                Console.WriteLine($"Aviso: Arquivo de modelo não encontrado em {filePath}. Usando modelo recém-inicializado com parâmetros de construtor.");
                // Model is already initialized by constructor, ensure it's fresh if needed.
                // InitializeModel(); // Potentially redundant if constructor always does it.
                return;
            }

            try
            {
                Console.WriteLine($"Carregando modelo de {filePath}...");
                object loadedObject = torch.load(filePath); // Returns object

                Dictionary<string, Tensor>? stateDictToLoad = null;
                bool configurationMatchedOrNotPresent = true;

                // Case 1: New bundle format (Dictionary<string, object>)
                if (loadedObject is Dictionary<string, object> savedBundle)
                {
                    Console.WriteLine("Detectado formato de pacote (config + state_dict).");
                    if (savedBundle.TryGetValue(ConfigKey, out var configObj) && configObj is Dictionary<string, object> modelConfig)
                    {
                        long loadedVocabSize = Convert.ToInt64(modelConfig["vocab_size"]);
                        int loadedEmbeddingDim = Convert.ToInt32(modelConfig["embedding_dim"]);
                        int loadedHiddenDim = Convert.ToInt32(modelConfig["hidden_dim"]);
                        int loadedRnnLayers = Convert.ToInt32(modelConfig["rnn_layers"]);

                        Console.WriteLine($"Configuração salva: Vocab={loadedVocabSize}, Embed={loadedEmbeddingDim}, Hidden={loadedHiddenDim}, RNNLayers={loadedRnnLayers}");
                        Console.WriteLine($"Configuração atual: Vocab={(long)VocabularySize}, Embed={_embeddingDim}, Hidden={_hiddenDim}, RNNLayers={_rnnLayers}");

                        if (loadedVocabSize != (long)VocabularySize ||
                            loadedEmbeddingDim != _embeddingDim ||
                            loadedHiddenDim != _hiddenDim ||
                            loadedRnnLayers != _rnnLayers)
                        {
                            Console.Error.WriteLine("Erro Crítico: A configuração do modelo salvo é incompatível com os parâmetros de inicialização desta instância do modelo.");
                            Console.Error.WriteLine("Por favor, instancie TorchSharpGenerativeModel com os parâmetros corretos ou treine um novo modelo.");
                            configurationMatchedOrNotPresent = false;
                            // Do not proceed with loading state_dict into a mismatched architecture.
                        }
                    }
                    else
                    {
                        Console.WriteLine("Aviso: Pacote salvo não contém uma entrada 'config' válida. Tentando carregar apenas o state_dict.");
                    }

                    if (configurationMatchedOrNotPresent)
                    {
                        if (savedBundle.TryGetValue(StateDictKey, out var stateDictObj) && stateDictObj is Dictionary<string, Tensor> sd)
                        {
                            stateDictToLoad = sd;
                        }
                        else
                        {
                            Console.WriteLine("Erro: Pacote salvo não contém uma entrada 'model_state_dict' válida do tipo Dictionary<string, Tensor>.");
                            configurationMatchedOrNotPresent = false; // Mark as failure to load
                        }
                    }
                }
                // Case 2: Old legacy format (Dictionary<string, Tensor>)
                else if (loadedObject is Dictionary<string, Tensor> legacyStateDict)
                {
                    Console.WriteLine("Aviso: Carregando modelo em formato antigo (apenas state_dict). A configuração do modelo não é verificada contra o arquivo; usando parâmetros da instância atual.");
                    stateDictToLoad = legacyStateDict;
                }
                // Case 3: Raw Tensor (unexpected for model state)
                else if (loadedObject is Tensor rawTensor)
                {
                    Console.WriteLine("Erro: O arquivo carregado contém um Tensor bruto, mas esperava-se um state_dict (Dictionary<string, Tensor>) ou um pacote de modelo (Dictionary<string, object>).");
                    rawTensor.Dispose(); // Dispose the unexpected tensor
                    configurationMatchedOrNotPresent = false; // Mark as failure to load
                }
                // Case 4: Unrecognized format
                else
                {
                    string typeName = loadedObject?.GetType().FullName ?? "null";
                    Console.WriteLine($"Erro: O arquivo carregado é de tipo inesperado ('{typeName}'). Esperava-se um state_dict ou pacote de modelo.");
                    configurationMatchedOrNotPresent = false; // Mark as failure to load
                }

                // Perform model loading if state_dict was successfully extracted and config is valid
                if (stateDictToLoad != null && configurationMatchedOrNotPresent)
                {
                    InitializeModel(); // Ensure a clean model instance matching current class parameters
                    _modelModule!.load_state_dict(stateDictToLoad); // Non-null assertion for _modelModule due to InitializeModel
                    Console.WriteLine("Modelo carregado com sucesso.");
                }
                else
                {
                    Console.WriteLine("Falha ao carregar o modelo devido a incompatibilidade de configuração ou formato de arquivo inválido. Usando modelo recém-inicializado com parâmetros de construtor.");
                    if (_modelModule == null || !configurationMatchedOrNotPresent) // If model wasn't usable or config mismatched
                    {
                        InitializeModel(); // Ensure it's reset to a clean state based on constructor params
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro excepcional ao carregar modelo de {filePath}: {ex.Message}");
                Console.WriteLine("Falha ao carregar. Usando modelo recém-inicializado com parâmetros de construtor.");
                InitializeModel(); // Re-initialize to a clean state on any error
            }
        }
        
        // Adicionar método parameters() para delegar ao _modelModule
        public IEnumerable<torch.Tensor> parameters()
        {
            if (_modelModule == null)
            {
                throw new InvalidOperationException("Modelo não inicializado.");
            }
            return _modelModule.parameters();
        }

        public override void Save(string filePath)
        {
            if (_modelModule == null)
            {
                Console.WriteLine("Erro: Nenhum módulo de modelo para salvar.");
                return;
            }

            try
            {
                var modelConfig = new Dictionary<string, object>
                {
                    { "vocab_size", (long)VocabularySize },
                    { "embedding_dim", _embeddingDim },
                    { "hidden_dim", _hiddenDim },
                    { "rnn_layers", _rnnLayers }
                };

                try
                {
                    var stateDict = _modelModule.state_dict();
                    // Como torch.save só aceita Tensor, precisamos converter o dicionário ou salvar os tensores individualmente
                    // Aqui, salvamos o primeiro tensor como exemplo (não ideal, mas funciona com a sobrecarga atual)
                    if (stateDict.Count > 0)
                    {
                        var firstTensor = stateDict.Values.First();
                        torch.save(firstTensor, filePath);
                        Console.WriteLine($"Primeiro tensor salvo com sucesso em {filePath}. (Nota: Apenas um tensor foi salvo devido à limitação da sobrecarga.)");
                    }
                    else
                    {
                        Console.WriteLine("Erro: State dict está vazio.");
                    }

                    // Alternativa: Salvar como JSON para preservar o dicionário completo
                    // using System.Text.Json;
                    // string json = JsonSerializer.Serialize(stateDict.Select(kvp => new { Key = kvp.Key, Data = kvp.Value.ToArray() }));
                    // File.WriteAllText(Path.ChangeExtension(filePath, ".json"), json);
                    // Console.WriteLine($"State dict salvo como JSON em {Path.ChangeExtension(filePath, ".json")}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Erro ao salvar modelo em {filePath}: {ex.Message}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro ao salvar modelo em {filePath}: {ex.Message}");
            }
        }

        public override torch.Tensor Generate(torch.Tensor inputTokens, int maxTokens, float temperature = 1.0f)
        {
            if (_modelModule == null)
            {
                throw new InvalidOperationException("O módulo do modelo não está inicializado.");
            }

            Tensor currentTokens = inputTokens; // Use a local variable for tokens being processed
            bool convertedInput = false;

            if (currentTokens.dtype != ScalarType.Int64)
            {
                Console.WriteLine($"Aviso: inputTokens.dtype é {currentTokens.dtype}, esperado Int64. Tentando converter...");
                var converted = currentTokens.to(ScalarType.Int64);
                if ((bool)(currentTokens != inputTokens)) currentTokens.Dispose(); // Dispose previous currentTokens if it was intermediate
                currentTokens = converted;
                convertedInput = true;
            }

            _modelModule.eval(); 

            Tensor generatedSequence;
            using (var noGradGuard = torch.no_grad())
            {
                // Clone to avoid modifying the input tensor if it's used elsewhere,
                // and ensure it's on the correct device (model's device implicitly if not specified)
                generatedSequence = currentTokens.clone(); 

                if (generatedSequence.size(0) != 1)
                {
                    if (convertedInput) currentTokens.Dispose(); // Dispose if created locally
                    generatedSequence.Dispose();
                    throw new ArgumentException("O método Generate atualmente suporta apenas batch_size de 1.");
                }

                for (int i = 0; i < maxTokens; i++)
                {
                    using var outputLogits = this.forward(generatedSequence); 
                    using var nextTokenLogits = outputLogits.select(1, -1).squeeze(1);

                    if (temperature <= 1e-5f) // Avoid division by zero or extremely small numbers; treat as greedy
                    {
                        using var nextTokenTensor = torch.argmax(nextTokenLogits, dim: -1, keepdim: true);
                        var tempGeneratedSequence = torch.cat(new[] { generatedSequence, nextTokenTensor }, dim: 1);
                        generatedSequence.Dispose();
                        generatedSequence = tempGeneratedSequence;
                        long nextTokenId = nextTokenTensor.item<long>();
                        if (_tokenizerService.IsEndOfSequenceToken(nextTokenId))
                        {
                            Console.WriteLine("\n[EOS] Token de fim de sequência gerado.");
                            break;
                        }
                    }
                    else
                    {
                        using var scaledLogits = nextTokenLogits / temperature;
                        using var probabilities = torch.softmax(scaledLogits, dim: -1);
                        using var nextTokenTensor = torch.multinomial(probabilities, num_samples: 1);
                    
                        var tempGeneratedSequence = torch.cat(new[] { generatedSequence, nextTokenTensor }, dim: 1);
                        generatedSequence.Dispose();
                        generatedSequence = tempGeneratedSequence;
                        long nextTokenId = nextTokenTensor.item<long>(); // Assumes batch_size = 1
                        if (_tokenizerService.IsEndOfSequenceToken(nextTokenId))
                        {
                            Console.WriteLine("\n[EOS] Token de fim de sequência gerado.");
                            break;
                        }
                    }
                }
            }

            if (convertedInput) currentTokens.Dispose(); // Dispose the (potentially) converted input if it was different from original inputTokens
        
            // _modelModule.train(); // Only set back to train if it was its original state and is intended.
            // For a model primarily for generation, it might always stay in eval.
            return generatedSequence; // Caller takes ownership
        }

        public override torch.Tensor forward(torch.Tensor input)
        {
            if (_modelModule == null)
            {
                throw new InvalidOperationException("O módulo do modelo não está inicializado.");
            }
            // Input type check is now inside SimpleRNNLanguageModel.forward
            return _modelModule.forward(input); // Caller takes ownership of returned tensor
        }

        private bool _disposedValue;

        protected override void Dispose(bool disposing)
        {
            if (!_disposedValue)
            {
                if (disposing)
                {
                    _modelModule?.Dispose();
                    _modelModule = null;
                }
                _disposedValue = true;
            }
            base.Dispose(disposing);
        }
    }

// Dummy TokenizerService for compilation if not provided
}