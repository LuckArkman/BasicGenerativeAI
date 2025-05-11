using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using BasicGenerativeAI.Services; // Assuming this exists and is correct
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using TorchSharp.Utils;

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
        private readonly SimpleRNNLanguageModel _modelModule;
        private readonly Device _device;
        private readonly int _vocabularySize;
        private bool _disposed;

        public TorchSharpGenerativeModel(TokenizerService tokenizerService, int device, long embeddingDim = 256, long hiddenDim = 512, long rnnLayers = 2)
        {
            if (tokenizerService == null)
            {
                throw new ArgumentNullException(nameof(tokenizerService));
            }

            _device = torch.CPU;
            _vocabularySize = tokenizerService.VocabularySize;
            _modelModule = new SimpleRNNLanguageModel(_vocabularySize, _device, (int)embeddingDim, (int)hiddenDim, (int)rnnLayers);
        }

        public override int VocabularySize => _vocabularySize;

        public override torch.Tensor Generate(torch.Tensor inputTokens, int maxTokens, float temperature = 1.0f)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(TorchSharpGenerativeModel));
            }

            using var logits = _modelModule.forward(inputTokens);
            // Implementação placeholder para geração
            return logits; // Substituir por lógica de amostragem com temperatura
        }

        public override torch.Tensor forward(torch.Tensor input)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(TorchSharpGenerativeModel));
            }

            return _modelModule.forward(input);
        }

        public override void Save(string path)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(TorchSharpGenerativeModel));
            }

            var stateDict = _modelModule.state_dict();
            if (stateDict.Count() == 0)
            {
                throw new InvalidOperationException("State dict está vazio.");
            }

            // Criar um diretório para salvar os tensores individualmente
            string directory = Path.GetDirectoryName(path) ?? ".";
            string baseName = Path.GetFileNameWithoutExtension(path);
            string tensorDir = Path.Combine(directory, $"{baseName}_tensors");
            Directory.CreateDirectory(tensorDir);

            // Salvar cada tensor em um arquivo separado
            var tensorFileMap = new Dictionary<string, string>();
            int tensorIndex = 0;
            foreach (var (name, tensor) in stateDict)
            {
                string tensorPath = Path.Combine(tensorDir, $"tensor_{tensorIndex}.pth");
                torch.save(tensor, tensorPath); // Salva o tensor individualmente
                tensorFileMap[name] = tensorPath;
                tensorIndex++;
            }

            // Salvar o mapeamento de nomes para arquivos em um arquivo de metadados
            string metadataPath = Path.Combine(directory, $"{baseName}_metadata.json");
            File.WriteAllText(metadataPath, JsonSerializer.Serialize(tensorFileMap));
        }

        public override void Load(string path)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(TorchSharpGenerativeModel));
            }

            string directory = Path.GetDirectoryName(path) ?? ".";
            string baseName = Path.GetFileNameWithoutExtension(path);
            string metadataPath = Path.Combine(directory, $"{baseName}_metadata.json");

            if (!File.Exists(metadataPath))
            {
                throw new FileNotFoundException("Arquivo de metadados não encontrado.", metadataPath);
            }

            var tensorFileMap = JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(metadataPath));
            if (tensorFileMap == null)
            {
                throw new InvalidOperationException("Não foi possível desserializar o arquivo de metadados.");
            }

            var loadedStateDict = new OrderedDict<string, torch.Tensor>();
            foreach (var (name, tensorPath) in tensorFileMap)
            {
                if (!File.Exists(tensorPath))
                {
                    throw new FileNotFoundException($"Arquivo de tensor não encontrado: {tensorPath}");
                }
                var tensorObj = torch.load(tensorPath);
                if (tensorObj is not torch.Tensor tensor)
                {
                    throw new InvalidOperationException($"O arquivo {tensorPath} não contém um tensor válido.");
                }
                loadedStateDict.Add(name, tensor);
            }

            // Converter OrderedDict para Dictionary para compatibilidade com load_state_dict
            var stateDictToLoad = new Dictionary<string, torch.Tensor>(loadedStateDict);
            _modelModule.load_state_dict(stateDictToLoad);
        }

        public IEnumerable<Parameter> parameters() // Corrigido para IEnumerable<Parameter>
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(TorchSharpGenerativeModel));
            }
            return _modelModule.parameters(); // Chama o método padrão de Module
        }

        protected override void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    _modelModule?.Dispose();
                }
                _disposed = true;
            }
        }
    }
}