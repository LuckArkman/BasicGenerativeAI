using HuggingFace.Tokenizers;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;

namespace BasicGenerativeAI.Services;

public class TokenizerService : IDisposable
    {
        private readonly Tokenizer _tokenizer;
        private readonly string _endOfSequenceToken; // Token especial para fim de sequência
        private readonly int _endOfSequenceTokenId;

        // Tamanho do vocabulário do tokenizer
        public int VocabularySize => (int)_tokenizer.Vocabulary.Count;

        // Construtor: Carrega o tokenizer GPT-2
        public TokenizerService()
        {
            // Carrega o tokenizer pré-treinado do GPT-2
            _tokenizer = Tokenizer.Create("gpt2"); // Ou path para arquivo local config.json

            // Identifica o token e ID de fim de sequência
            // O token EOS padrão para GPT-2 é <|endoftext|>
            _endOfSequenceToken = "<|endoftext|>";
            // O ID pode variar dependendo da versão do tokenizer, mas para GPT-2 base é geralmente 50256
             // Vamos buscar o ID no vocabulário para garantir.
             var vocab = _tokenizer.Vocabulary;
             if(vocab.TryGetValue(_endOfSequenceToken, out uint eosId))
             {
                 _endOfSequenceTokenId = (int)eosId;
             }
             else
             {
                 // Fallback ou erro se EOS não for encontrado (improvável para gpt2)
                 Console.WriteLine($"Aviso: EOS token '{_endOfSequenceToken}' não encontrado no vocabulário. Usando 50256 como fallback.");
                 _endOfSequenceTokenId = 50256;
             }


            Console.WriteLine($"Tokenizer GPT-2 carregado. Vocabulário: {VocabularySize} tokens. EOS Token ID: {_endOfSequenceTokenId}");
        }

        // Tokeniza uma string em uma lista de IDs
        public List<long> Encode(string text)
        {
            // O HuggingFace.Tokenizers retorna um Encoding, que contém IDs e outras infos.
            Tokenizers.Encoding encoding = _tokenizer.Encode(text);
            // Convertendo uint[] para List<long> para compatibilidade com TorchSharp Tensor (Int64)
            return encoding.Ids.Select(id => (long)id).ToList();
        }

        // Converte uma lista de IDs de volta para uma string
        public string Decode(IEnumerable<long> tokenIds)
        {
             // O Decode espera uint[], então convertemos de volta.
            uint[] ids = tokenIds.Select(id => (uint)id).ToArray();
            return _tokenizer.Decode(ids);
        }

        // Converte uma lista de IDs em um Tensor Long de TorchSharp
        public torch.Tensor EncodeToTensor(string text)
        {
            var ids = Encode(text);
            // Cria um tensor Long (Int64) a partir dos IDs
            // O shape esperado pelo modelo pode ser (batch_size, sequence_length)
            // Para um único input, shape (1, sequence_length)
            return torch.tensor(ids.ToArray(), dtype: torch.ScalarType.Int64).unsqueeze(0); // Adiciona dimensão de batch (size 1)
        }

         // Verifica se um token ID é o token de fim de sequência
        public bool IsEndOfSequenceToken(long tokenId)
        {
            return tokenId == _endOfSequenceTokenId;
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
                    // O objeto Tokenizer HuggingFace.Tokenizers precisa ser explicitamente Disposed?
                    // A documentação sugere que sim, se ele detém recursos nativos.
                    _tokenizer.Dispose();
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