using BasicGenerativeAI.Core;
using BasicGenerativeAI.Services;
using TorchSharp; // Para Tensor
using System.Threading.Tasks;
using System.Linq;
using System.Collections.Generic;

namespace BasicGenerativeAI.System;

// Gerencia o fluxo do diálogo, integrando o modelo, tokenizer, histórico e busca.
    public class DialogManager : IDisposable
    {
        private readonly BaseGenerativeModel _model; // Usa a interface base (polimorfismo)
        private readonly TokenizerService _tokenizerService;
        private readonly ConversationHistory _history;
        private readonly GoogleSearchService? _searchService; // Pode ser opcional
        private readonly int _maxResponseTokens = 100; // Limite de tokens para a resposta

        // Construtor
        public DialogManager(BaseGenerativeModel model, TokenizerService tokenizerService,
                             ConversationHistory history, GoogleSearchService? searchService = null)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _tokenizerService = tokenizerService ?? throw new ArgumentNullException(nameof(tokenizerService));
            _history = history ?? throw new ArgumentNullException(nameof(history));
            _searchService = searchService; // Pode ser null

            Console.WriteLine($"DialogManager inicializado. Modelo: {_model.GetType().Name}");
        }

        // Processa a entrada do usuário, gera uma resposta da AI.
        public async Task<string> ProcessInputAsync(string userInput)
{
    if (string.IsNullOrWhiteSpace(userInput))
    {
        return "Por favor, digite algo.";
    }

    // Adiciona a fala do usuário ao histórico
    _history.AddTurn("User", userInput);

    string contextForModel = _history.GetFormattedHistory();

    // --- Lógica para verificar e realizar busca na internet ---
    if (_searchService != null && userInput.Trim().StartsWith("Buscar por ", StringComparison.OrdinalIgnoreCase))
    {
        var searchQuery = userInput.Trim().Substring("Buscar por ".Length).Trim();
        Console.WriteLine($"Detectada intenção de busca: '{searchQuery}'");
        _history.AddTurn("System", $"Realizando busca por '{searchQuery}'...");

        var searchResults = await _searchService.SearchAsync(searchQuery);

        var searchResultsText = $"Resultados da busca para '{searchQuery}':\n" + string.Join("\n---\n", searchResults);
        _history.AddTurn("System", searchResultsText);
        contextForModel = _history.GetFormattedHistory();
    }
    // --- Fim da lógica de busca ---

    Console.WriteLine("Contexto para o modelo:\n---\n" + contextForModel + "\n---");

    // Tokeniza o histórico formatado
    var (inputIds, attentionMask) = _tokenizerService.EncodeToTensor(contextForModel);
    Console.WriteLine($"Input tensor shape: [{string.Join(", ", inputIds.shape)}]. Num tokens: {inputIds.size(1)}");

    // Gera a resposta usando o modelo
    using var outputTensor = _model.Generate(inputIds, _maxResponseTokens);
    Console.WriteLine($"Output tensor shape: [{string.Join(", ", outputTensor.shape)}]. Num tokens: {outputTensor.size(1)}");

    // Decodifica os tokens gerados
    var squeezedTensor = outputTensor.squeeze(0); // Remove a dimensão de batch
    var tokenIds = squeezedTensor.data<long>().ToArray(); // Converte o tensor para array de long
    var generatedTokenIds = tokenIds.Skip((int)inputIds.size(1)).ToList(); // Pula os tokens de entrada
    string generatedText = _tokenizerService.Decode(generatedTokenIds, skipSpecialTokens: true);

    // Limpa tokens especiais e espaços extras
    generatedText = generatedText
        .Replace(_tokenizerService.EndOfSequenceTokenString, "")
        .Replace(_tokenizerService.Decode(new[] { (long)_tokenizerService.PadTokenId }), "")
        .Trim();

    // Remove prefixos indesejados
    if (generatedText.StartsWith("AI: ", StringComparison.OrdinalIgnoreCase))
    {
        generatedText = generatedText.Substring("AI: ".Length).Trim();
    }
    if (generatedText.StartsWith("AI:", StringComparison.OrdinalIgnoreCase))
    {
        generatedText = generatedText.Substring("AI:".Length).Trim();
    }

    // Adiciona a resposta ao histórico
    _history.AddTurn("AI", generatedText);

    // Dispor tensores
    inputIds.Dispose();
    attentionMask.Dispose();
    squeezedTensor.Dispose();
    outputTensor.Dispose();

    return generatedText;
}

        // Salva o estado do modelo
        public void SaveModel(string filePath)
        {
            _model.Save(filePath);
        }

        // Carrega o estado do modelo
        public void LoadModel(string filePath)
        {
            _model.Load(filePath);
        }

        // Implementação de IDisposable
        private bool disposedValue;

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // Dispor recursos gerenciados
                    _model.Dispose();
                    _tokenizerService.Dispose();
                    _searchService?.Dispose(); // Dispor se não for null
                    // O histórico não precisa de Dispose explícito
                }

                // Dispor recursos não gerenciados
                disposedValue = true;
            }
        }

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }