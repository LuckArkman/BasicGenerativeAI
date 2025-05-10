// BasicGenerativeAI/Program.cs (Atualizado)

using BasicGenerativeAI;
using BasicGenerativeAI.Core;
using BasicGenerativeAI.Services;
using BasicGenerativeAI.System;
using Microsoft.Extensions.Configuration;
using TorchSharp; // Para Device

// Configurar o carregamento do appsettings.json
var builder = new ConfigurationBuilder()
    .SetBasePath(Directory.GetCurrentDirectory())
    .AddJsonFile("appsettings.json", optional: true, reloadOnChange: true);

IConfiguration configuration = builder.Build();

// Obter configurações da API do Google
var googleApiKey = configuration["GoogleSearch:ApiKey"];
var googleCx = configuration["GoogleSearch:Cx"];

// Caminho para salvar/carregar o modelo
var modelSavePath = "basic_ai_model_state.pth"; // Extensão .pth é comum para arquivos PyTorch/TorchSharp state_dict

// Configurações de treinamento (exemplo)
var trainingEpochs = 50;
var trainingBatchSize = 8; // Tamanho do batch para treinamento
var maxSequenceLength = 64; // Comprimento das sequências de treinamento
var learningRate = 0.001;
var embeddingDim = 256; // Deve ser o mesmo usado no modelo
var hiddenDim = 512;    // Deve ser o mesmo usado no modelo
var rnnLayers = 2;      // Deve ser o mesmo usado no modelo

// Configurar dispositivo (CPU ou GPU)
// Use Device.CUDA if CUDA is available and you installed TorchSharp-cuda-xxx package
var device = torch.CPU;
if (torch.cuda.is_available())
{
    device = torch.CUDA;
    Console.WriteLine("CUDA disponível. Usando GPU para treinamento/inferência.");
}
else
{
    Console.WriteLine("CUDA não disponível. Usando CPU para treinamento/inferência.");
}


Console.WriteLine("Inicializando AI Generativa Básica...");

// Use blocos using para garantir que os recursos TorchSharp sejam dispostos
using (var tokenizerService = new TokenizerService())
{
    // Inicializa o modelo usando a implementação TorchSharp
    // Passamos o TokenizerService e os parâmetros da arquitetura
    using (var model = new TorchSharpGenerativeModel(tokenizerService, embeddingDim, hiddenDim, rnnLayers))
    {
        // Verifica se deve treinar
        Console.Write("Deseja treinar o modelo antes de iniciar? (s/n, 's' para treinar): ");
        var trainChoice = Console.ReadLine()?.ToLower();

        if (trainChoice == "s")
        {
             // Instancia o script de treinamento
             var trainingScript = new TrainingScript(tokenizerService, model,
                                                       trainingEpochs, trainingBatchSize, maxSequenceLength, learningRate,
                                                       modelSavePath, device);
             trainingScript.RunTraining();

             // Após o treinamento, o modelo já foi salvo e está carregado na memória
             // Se você quisesse carregar o modelo salvo, faria:
             // model.Load(modelSavePath);
        }
        else
        {
            // Se não treinar, tenta carregar um modelo salvo existente
            model.Load(modelSavePath);
        }


        // Use blocos using para outros componentes
        using (var history = new ConversationHistory(maxTurns: 1000))
        {
            // Inicializa o serviço de busca *apenas* se as configurações estiverem presentes
            using (var searchService = (string.IsNullOrEmpty(googleApiKey) || string.IsNullOrEmpty(googleCx)) ?
                                        null : new GoogleSearchService(googleApiKey, googleCx))
            {
                // Inicializa o gerenciador de diálogo com todos os componentes
                using (var dialogManager = new DialogManager(model, tokenizerService, history, searchService))
                {
                    Console.WriteLine("\nAI pronta. Digite sua mensagem (ou 'sair' para fechar, 'salvar' para salvar o modelo, 'limpar' para limpar histórico).");
                    if (searchService != null)
                    {
                        Console.WriteLine("Experimente 'Buscar por [sua pergunta]' para usar a busca.");
                    }
                    else
                    {
                         Console.WriteLine("Serviço de busca do Google não configurado (adicione API Key e CX em appsettings.json).");
                    }


                    // Loop principal de interação
                    while (true)
                    {
                        Console.Write("Você: ");
                        var input = Console.ReadLine();

                        if (string.IsNullOrWhiteSpace(input))
                        {
                            continue;
                        }

                        if (input.ToLower() == "sair")
                        {
                            break; // Sair do loop
                        }

                        if (input.ToLower() == "salvar")
                        {
                            dialogManager.SaveModel(modelSavePath);
                            continue; // Voltar para o input
                        }

                        if (input.ToLower() == "limpar")
                        {
                            history.Clear();
                            Console.WriteLine("Histórico da conversa limpo.");
                            continue; // Voltar para o input
                        }

                        // Processa o input do usuário e obtém a resposta da AI
                        string aiResponse = await dialogManager.ProcessInputAsync(input);

                        Console.WriteLine($"AI: {aiResponse}");
                    }

                    // Opcional: Salvar o modelo ao sair
                    Console.Write("Salvar o estado atual do modelo antes de sair? (s/n): ");
                    if (Console.ReadLine()?.ToLower() == "s")
                    {
                        dialogManager.SaveModel(modelSavePath);
                    }
                }
            }
        }
    } // Dispose do modelo
} // Dispose do tokenizer

Console.WriteLine("AI encerrada.");