using BasicGenerativeAI.Core; // Para SimpleBPETokenizer
using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp; // Necessário para torch.Tensor
using static TorchSharp.torch; // Necessário para torch.Tensor

namespace BasicGenerativeAI.Services
{

    public class TokenizerService : IDisposable
    {
        public int VocabularySize { get; private set; } = 1000; // Example value
        public long EOSTokenId { get; private set; } = 2; // Example EOS token ID
        public Tensor Encode(string text) => torch.randint(0, VocabularySize, new long[] {1, 10}, dtype: torch.ScalarType.Int64);
        public string Decode(torch.Tensor tokenIds) => "decoded text example";
        private readonly SimpleBPETokenizer _tokenizer; // Usando SimpleBPETokenizer interno
        private readonly string _endOfSequenceTokenString;
        private readonly int _endOfSequenceTokenId;
        private readonly string _padTokenString;
        private readonly int _padTokenId;
        public int EndOfSequenceTokenId => _endOfSequenceTokenId;
        public int PadTokenId => _padTokenId;
        public string EndOfSequenceTokenString => _endOfSequenceTokenString;

        public TokenizerService() // Remover o parâmetro
        {
            _tokenizer = new SimpleBPETokenizer();

            var corpus = new List<string>
            {
                "hello world", "this is a test", "basic generative ai", "how are you", "machine learning"
            };
            _tokenizer.Train(corpus, numMerges: 50);

            _endOfSequenceTokenString = "<|endoftext|>";
            if (!_tokenizer.Vocab.ContainsKey(_endOfSequenceTokenString))
            {
                _tokenizer.Vocab[_endOfSequenceTokenString] = _tokenizer.VocabularySize;
                _tokenizer.InverseVocab[_tokenizer.VocabularySize] = _endOfSequenceTokenString;
            }

            _endOfSequenceTokenId = _tokenizer.Vocab[_endOfSequenceTokenString];

            _padTokenString = "<PAD>";
            if (!_tokenizer.Vocab.ContainsKey(_padTokenString))
            {
                _tokenizer.Vocab[_padTokenString] = _tokenizer.VocabularySize;
                _tokenizer.InverseVocab[_tokenizer.VocabularySize] = _padTokenString;
            }

            _padTokenId = _tokenizer.Vocab[_padTokenString];

            Console.WriteLine(
                $"Tokenizer interno carregado. Vocabulário: {VocabularySize}. EOS ID: {_endOfSequenceTokenId}. PAD ID: {_padTokenId}.");
        }

        public long[] TokenizeText(string text)
        {
            if (string.IsNullOrEmpty(text))
                return Array.Empty<long>();

            return Encode(text, addSpecialTokens: true).ToArray(); // Usar Encode para incluir EOS
        }

        public List<long> Encode(string text, bool addSpecialTokens = true, int? maxLength = null)
        {
            if (string.IsNullOrEmpty(text))
                return new List<long>();

            var encoded = _tokenizer.Encode(text);

            if (addSpecialTokens)
            {
                encoded.Add(_endOfSequenceTokenId);
            }

            if (maxLength.HasValue)
            {
                if (encoded.Count > maxLength.Value)
                {
                    encoded = encoded.Take(maxLength.Value).ToList();
                }
                else if (encoded.Count < maxLength.Value)
                {
                    encoded.AddRange(Enumerable.Repeat(_padTokenId, maxLength.Value - encoded.Count));
                }
            }

            return encoded.Select(id => (long)id).ToList();
        }

        public string Decode(IEnumerable<long> tokenIds, bool skipSpecialTokens = true)
        {
            if (tokenIds == null) return string.Empty;

            var ids = tokenIds.Select(id => (int)id).ToList();
            if (skipSpecialTokens)
            {
                ids = ids.Where(id => id != _padTokenId && id != _endOfSequenceTokenId).ToList();
            }

            return _tokenizer.Decode(ids);
        }

        public (torch.Tensor inputIds, torch.Tensor attentionMask) EncodeToTensors(string text,
            bool addSpecialTokens = true, int? maxLength = null)
        {
            if (string.IsNullOrEmpty(text))
                return (torch.tensor(new long[0], dtype: torch.ScalarType.Int64),
                    torch.tensor(new long[0], dtype: torch.ScalarType.Int64));

            var encoded = Encode(text, addSpecialTokens, maxLength);
            var attentionMask = encoded.Select(id => id == _padTokenId ? 0L : 1L).ToArray();

            var inputIdsTensor = torch.tensor(encoded.ToArray(), dtype: torch.ScalarType.Int64).unsqueeze(0);
            var attentionMaskTensor = torch.tensor(attentionMask, dtype: torch.ScalarType.Int64).unsqueeze(0);

            return (inputIdsTensor, attentionMaskTensor);
        }

        public (torch.Tensor inputIds, torch.Tensor attentionMask) EncodeBatchToTensors(
            List<string> texts,
            bool addSpecialTokens = true,
            int? maxLength = null,
            bool padToLongestInBatchIfNotMaxLength = true)
        {
            if (texts == null || !texts.Any())
                throw new ArgumentException("A lista de textos não pode ser nula ou vazia.");

            var encodedBatch = texts.Select(t => Encode(t, addSpecialTokens, null)).ToList();
            int maxLenInBatch = maxLength ?? (padToLongestInBatchIfNotMaxLength
                ? encodedBatch.Max(e => e.Count)
                : encodedBatch.First().Count);

            long[,] ids2D = new long[encodedBatch.Count, maxLenInBatch];
            long[,] attentionMask2D = new long[encodedBatch.Count, maxLenInBatch];

            for (int i = 0; i < encodedBatch.Count; i++)
            {
                var encoded = encodedBatch[i];
                for (int j = 0; j < maxLenInBatch; j++)
                {
                    if (j < encoded.Count)
                    {
                        ids2D[i, j] = encoded[j];
                        attentionMask2D[i, j] = 1;
                    }
                    else
                    {
                        ids2D[i, j] = _padTokenId;
                        attentionMask2D[i, j] = 0;
                    }
                }
            }

            var inputIdsTensor = torch.tensor(ids2D, dtype: torch.ScalarType.Int64);
            var attentionMaskTensor = torch.tensor(attentionMask2D, dtype: torch.ScalarType.Int64);

            return (inputIdsTensor, attentionMaskTensor);
        }

        public bool IsEndOfSequenceToken(long tokenId) => tokenId == _endOfSequenceTokenId;
        public bool IsPadToken(long tokenId) => tokenId == _padTokenId;

        private bool disposedValue;

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // SimpleBPETokenizer não requer Dispose, mas mantemos a estrutura para compatibilidade futura
                }

                disposedValue = true;
            }
        }

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }

        ~TokenizerService() => Dispose(disposing: false);
    }
}