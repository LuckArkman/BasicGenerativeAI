using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using BasicGenerativeAI.Services;
using System.Collections.Generic; // Necessário para tipos usados nos parâmetros do modelo

namespace BasicGenerativeAI.Core;

public class TorchSharpGenerativeModel : BaseGenerativeModel
    {
        // O módulo neural TorchSharp que representa o modelo
        private Module<Tensor, Tensor>? modelModule;

        // Referência ao TokenizerService para obter o tamanho do vocabulário
        private readonly TokenizerService _tokenizerService;

        public override int VocabularySize => _tokenizerService.VocabularySize;

        // Parâmetros do modelo (exemplo) - você precisaria ajustar/adicionar mais
        private readonly int _embeddingDim; // Dimensão dos embeddings de token
        private readonly int _hiddenDim;    // Dimensão dos estados ocultos do RNN
        private readonly int _rnnLayers;    // Número de camadas RNN

        // Construtor
        public TorchSharpGenerativeModel(TokenizerService tokenizerService, int embeddingDim = 256, int hiddenDim = 512, int rnnLayers = 2)
        {
            _tokenizerService = tokenizerService ?? throw new ArgumentNullException(nameof(tokenizerService));
            _embeddingDim = embeddingDim;
            _hiddenDim = hiddenDim;
            _rnnLayers = rnnLayers;

            // Inicializa o módulo com a arquitetura definida
            modelModule = new SimpleRNNLanguageModel(
                vocabSize: VocabularySize,
                embeddingDim: _embeddingDim,
                hiddenDim: _hiddenDim,
                rnnLayers: _rnnLayers
            );

            Console.WriteLine($"Modelo TorchSharp inicializado com Embedding Dim: {_embeddingDim}, Hidden Dim: {_hiddenDim}, RNN Layers: {_rnnLayers}");
        }

        // Classe interna: Uma arquitetura de Language Model RNN simples
        private class SimpleRNNLanguageModel : Module<Tensor, Tensor>
        {
            private Embedding embedding;
            private GRU rnn; // Usando GRU, mais simples que LSTM, mas similar
            private Linear linear; // Layer final para mapear hidden state para logits

            public SimpleRNNLanguageModel(long vocabSize, int embeddingDim, int hiddenDim, int rnnLayers) : base("SimpleRNNLanguageModel")
            {
                // Layer de Embedding: mapeia IDs de token para vetores densos
                embedding = Embedding(vocabSize, embeddingDim);

                // Layer GRU: processa a sequência. batch_first = true significa input/output (batch, seq, feature)
                rnn = GRU(embeddingDim, hiddenDim, rnnLayers, batchFirst: true);

                // Layer Linear: mapeia a saída do RNN para o tamanho do vocabulário (logits)
                linear = Linear(hiddenDim, vocabSize);

                this.RegisterComponents(); // Registra sub-módulos para que o .parameters() e .to() funcionem
            }

            // O forward pass do modelo
            // input: Tensor de IDs de token, shape (batch_size, sequence_length)
            // hidden: Estado oculto inicial (opcional), shape (num_layers, batch_size, hidden_dim)
            public override Tensor forward(Tensor input, Tensor? hidden = null)
            {
                // 1. Embedding: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
                var embedded = embedding.forward(input);

                // 2. RNN: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, hidden_dim), (num_layers, batch_size, hidden_dim)
                // A saída 'output' contém o hidden state para CADA passo de tempo na sequência
                var (output, h_n) = rnn.forward(embedded, hidden);

                // 3. Linear: mapeia os hidden states de cada passo de tempo para logits
                // (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, vocab_size)
                // Precisamos aplicar a layer linear a cada passo de tempo.
                // Podemos "achatar" as duas primeiras dimensões temporariamente:
                // (batch_size * seq_len, hidden_dim) -> (batch_size * seq_len, vocab_size)
                var reshaped_output = output.view(output.size(0) * output.size(1), output.size(2));
                var logits = linear.forward(reshaped_output);

                // Reformatar de volta para (batch_size, seq_len, vocab_size)
                logits = logits.view(output.size(0), output.size(1), logits.size(1));

                // Não precisamos do estado oculto final (h_n) para a geração passo a passo neste setup simples
                // mas ele é retornado pelo GRU.

                return logits; // Retorna logits para cada token na sequência, para prever o próximo token
            }
        }


        // Carrega os pesos do modelo (state_dict) de um arquivo
        public override void Load(string filePath)
        {
            if (modelModule == null)
            {
                 // Isso não deve acontecer com a inicialização no construtor, mas para segurança
                 Console.WriteLine("Erro interno: modelModule é null ao tentar carregar.");
                 modelModule = new SimpleRNNLanguageModel(VocabularySize, _embeddingDim, _hiddenDim, _rnnLayers);
            }

            if (!File.Exists(filePath))
            {
                Console.WriteLine($"Aviso: Arquivo de modelo não encontrado em {filePath}. Inicializando um módulo novo.");
                 // Se o arquivo não existe, apenas garante que o módulo foi inicializado no construtor.
                 // Não sobrescreve com um módulo novo, apenas se modelModule for null.
                 if(modelModule == null) modelModule = new SimpleRNNLanguageModel(VocabularySize, _embeddingDim, _hiddenDim, _rnnLayers);
                 return;
            }

            try
            {
                Console.WriteLine($"Carregando modelo de {filePath}...");
                 // Carrega apenas o state_dict (pesos)
                 using var state_dict = torch.load(filePath);

                 // Cria um novo módulo para garantir que a estrutura corresponde ao state_dict
                 // Em um cenário real, você verificaria se a estrutura é compatível.
                 modelModule?.Dispose(); // Dispor o módulo antigo
                 modelModule = new SimpleRNNLanguageModel(VocabularySize, _embeddingDim, _hiddenDim, _rnnLayers);

                 modelModule.load_state_dict(state_dict);
                 Console.WriteLine("Modelo carregado com sucesso (state_dict).");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro ao carregar modelo de {filePath}: {ex.Message}");
                Console.WriteLine("Inicializando um novo modelo vazio.");
                 // Em caso de erro, inicializar um módulo novo e vazio
                 modelModule?.Dispose();
                modelModule = new SimpleRNNLanguageModel(VocabularySize, _embeddingDim, _hiddenDim, _rnnLayers);
            }
        }

        // Salva os pesos do modelo (state_dict) em um arquivo
        public override void Save(string filePath)
        {
            if (modelModule == null)
            {
                Console.WriteLine("Erro: Nenhum módulo de modelo para salvar.");
                return;
            }

            try
            {
                // Salva apenas o state_dict (pesos e biases)
                torch.save(modelModule.state_dict(), filePath);
                Console.WriteLine($"Modelo salvo com sucesso em {filePath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro ao salvar modelo em {filePath}: {ex.Message}");
            }
        }

         // Gera uma sequência de tokens a partir do input (implementação autoregressiva).
        public override torch.Tensor Generate(torch.Tensor inputTokens, int maxTokens, float temperature = 1.0f)
        {
            if (modelModule == null)
            {
                throw new InvalidOperationException("Model module is not initialized.");
            }

            // Garante que o modelo esteja em modo de avaliação (inference)
            modelModule.eval();

            // Garante que não haja cálculo de gradientes durante a inferência
            using var noGradGuard = torch.no_grad();

            // Clona o tensor de input para não modificá-lo
            var generatedTokens = inputTokens.clone(); // Shape (batch_size, current_seq_len)

            // Assume batch_size = 1 para este exemplo simples
            if (generatedTokens.size(0) != 1)
            {
                 throw new ArgumentException("Generate method currently only supports batch_size of 1.");
            }


            // Loop para gerar tokens um por um
            for (int i = 0; i < maxTokens; i++)
            {
                 // Para geração autoregressiva, passamos a sequência completa gerada *até agora*
                 // para o modelo e prevemos o PRÓXIMO token.
                 // O modelo RNN SimpleRNNLanguageModel foi projetado para receber (batch, seq_len)
                 // e retornar logits para cada posição na sequência.
                 // Para prever o próximo token, só precisamos dos logits da *última* posição.

                 Tensor outputLogitsAllSteps = modelModule.forward(generatedTokens); // Shape (1, current_seq_len, vocab_size)

                 // Pega os logits apenas para o *último* token da sequência
                 // Shape (1, 1, vocab_size) -> squeeze(1) -> Shape (1, vocab_size)
                 using var nextTokenLogits = outputLogitsAllSteps.slice(dim: 1, start: generatedTokens.size(1) - 1, end: generatedTokens.size(1)).squeeze(1);


                // Aplicar temperatura e softmax para obter probabilidades
                Tensor probabilities = (nextTokenLogits / temperature).softmax(dim: 1); // softmax na dimensão do vocabulário (dim=1)

                // Amostragem: Escolher o próximo token com base nas probabilidades
                // O torch.multinomial espera input (batch_size, num_categories)
                // probabilities já está no shape (1, vocab_size)
                Tensor nextTokenTensor = torch.multinomial(probabilities, num_samples: 1); // Shape (1, 1) -> representando o ID do próximo token

                long nextTokenId = nextTokenTensor.item<long>(); // Extrai o ID como long

                // Concatena o novo token gerado com a sequência existente
                // nextTokenTensor é shape (1, 1), generatedTokens é shape (1, current_seq_len)
                generatedTokens = torch.cat(new[] { generatedTokens, nextTokenTensor }, dim: 1); // Concatena ao longo da dimensão da sequência


                // Verificar se o token gerado é o token de fim de sequência
                if (_tokenizerService.IsEndOfSequenceToken(nextTokenId))
                {
                    Console.WriteLine("\n[EOS] Token de fim de sequência gerado.");
                    break; // Parar a geração se EOS for atingido
                }

                // Dispor tensores temporários criados dentro do loop
                probabilities.Dispose();
                nextTokenTensor.Dispose();
                nextTokenLogits.Dispose();
                outputLogitsAllSteps.Dispose(); // Dispor a saída do forward pass
            }

            // Retorna a sequência completa gerada (input original + generated)
            // Você pode querer retornar apenas os tokens *novos* gerados.
            // Para isso, salve o tamanho inicial do inputTokens e slize generatedTokens.
            // Exemplo: var initialLen = inputTokens.size(1); return generatedTokens.slice(dim: 1, start: initialLen, end: generatedTokens.size(1));

            // Para simplificar, retornamos a sequência completa que inclui o input original
            return generatedTokens;
        }

        // Implementação de Dispose para liberar recursos TorchSharp
        private bool disposedValue; // Garantir que dispose não seja chamado múltiplas vezes

        protected override void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // Dispor recursos gerenciados (se houver)
                }

                // Dispor o módulo TorchSharp
                modelModule?.Dispose();
                modelModule = null; // Garantir que a referência seja nula

                disposedValue = true;
            }
            base.Dispose(disposing); // Chamar o Dispose da base
        }

         // A implementação da interface IDisposable chama Dispose(true)
         // Não precisamos de um finalizador aqui pois BaseGenerativeModel já tem um padrão.
    }