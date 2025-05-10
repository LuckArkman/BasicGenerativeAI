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
            // Exemplo simples: Se a pergunta começar com "Buscar por", realizar a busca.
            if (_searchService != null && userInput.Trim().StartsWith("Buscar por ", StringComparison.OrdinalIgnoreCase))
            {
                var searchQuery = userInput.Trim().Substring("Buscar por ".Length).Trim();
                Console.WriteLine($"Detectada intenção de busca: '{searchQuery}'");
                _history.AddTurn("System", $"Realizando busca por '{searchQuery}'..."); // Adiciona nota no histórico para o modelo

                var searchResults = await _searchService.SearchAsync(searchQuery);

                // Formata os resultados da busca e adiciona ao contexto/histórico para o modelo
                var searchResultsText = $"Resultados da busca para '{searchQuery}':\n" + string.Join("\n---\n", searchResults);

                // Decide como incorporar os resultados:
                // Opção 1: Adicionar como um turno do sistema no histórico. O modelo verá "System: Resultados..."
                _history.AddTurn("System", searchResultsText);
                contextForModel = _history.GetFormattedHistory(); // Re-gera o contexto com os resultados

                // Opção 2 (Alternativa): Pre-pender os resultados ao input atual *sem* adicionar no histórico principal
                // (Pode ser útil se não quer poluir o histórico, mas quer que o modelo use a info *agora*)
                // string contextForModel = searchResultsText + "\n\n" + _history.GetFormattedHistory();
            }
             // --- Fim da lógica de busca ---


            Console.WriteLine("Contexto para o modelo:\n---\n" + contextForModel + "\n---");

            // Tokeniza o histórico formatado
            using var inputTensor = _tokenizerService.EncodeToTensor(contextForModel);
            Console.WriteLine($"Input tensor shape: {inputTensor.shape}. Num tokens: {inputTensor.size(1)}");


            // Gera a resposta usando o modelo
            // O método Generate retornará inputTokens + tokens gerados
            using var outputTensor = _model.Generate(inputTensor, _maxResponseTokens);
            Console.WriteLine($"Output tensor shape: {outputTensor.shape}. Num tokens: {outputTensor.size(1)}");

            // Decodifica os tokens gerados de volta para texto
            // Precisamos pegar apenas os tokens *novos* gerados.
            // O inputTensor original tinha size(1) tokens. O outputTensor tem mais.
            // Os tokens gerados começam após o tamanho do input original.
             var generatedTokenIds = outputTensor.squeeze(0).Skip((int)inputTensor.size(1)).ToList(); // Remove batch dim e pula tokens de input
            string generatedText = _tokenizerService.Decode(generatedTokenIds);

            // O decode pode incluir o token de fim de sequência. Podemos querer removê-lo.
             generatedText = generatedText.Replace(_tokenizerService.Decode(new[] { (long)_tokenizerService.VocabularySize -1 }), ""); // Remove </|endoftext|> - ID 50256 para gpt2 base
             generatedText = generatedText.Replace(_tokenizerService.Decode(new[] { (long)_tokenizerService.VocabularySize }), ""); // Remove o próximo (se existir)
             generatedText = generatedText.Trim(); // Limpa espaços em branco

            // Adiciona a resposta da AI ao histórico
             // Adiciona apenas a parte da resposta *após* o "AI:" que foi incluído no prompt.
             // Remove o prefixo "AI:" se o decode o gerou.
             if(generatedText.StartsWith("AI: ", StringComparison.OrdinalIgnoreCase))
             {
                 generatedText = generatedText.Substring("AI: ".Length).Trim();
             }
             // Remove a parte do prompt "AI:" se ela foi gerada acidentalmente
             var historyPrefix = "AI:";
             if(generatedText.StartsWith(historyPrefix, StringComparison.OrdinalIgnoreCase))
             {
                 generatedText = generatedText.Substring(historyPrefix.Length).Trim();
             }


            _history.AddTurn("AI", generatedText);

            // Dispor tensores que não são mais necessários
             inputTensor.Dispose();
             outputTensor.Dispose(); // generate() retorna um novo tensor, então o tensor original é seguro

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