using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace BasicGenerativeAI.Core;

public class SimpleRNNLanguageModel : Module<Tensor, Tensor>
{
    private readonly Embedding _embedding;
    private readonly GRU _rnn;
    private readonly Linear _linear;

    public SimpleRNNLanguageModel(long vocabSize, int embeddingDim, int hiddenDim, int rnnLayers) : base(
        "SimpleRNNLanguageModel")
    {
        _embedding = Embedding(vocabSize, embeddingDim);
        _rnn = GRU(inputSize: embeddingDim, hiddenSize: hiddenDim, numLayers: rnnLayers, batchFirst: true);
        _linear = Linear(hiddenDim, vocabSize);

        RegisterComponents();
    }

    public Tensor forward(Tensor input, Tensor? hidden = null)
    {
        if (input.dtype != ScalarType.Int64)
        {
            // This should ideally be caught earlier or ensured by the caller.
            // For robustness, one might cast, but it's often better to fix the source.
            string shapeStr = $"[{string.Join(", ", input.shape)}]";
            throw new ArgumentException(
                $"Input tensor for embedding must be Int64, but got {input.dtype}. Original shape: {shapeStr}");
        }

        using var embedded = _embedding.forward(input);
        var rnnOutputTuple = _rnn.forward(embedded, hidden);
        using var output = rnnOutputTuple.Item1;
        // rnnOutputTuple.Item2 is the final hidden state, dispose if not used or returned
        rnnOutputTuple.Item2?.Dispose();


        using var reshapedOutput = output.reshape(-1, output.size(2));
        using var logits = _linear.forward(reshapedOutput);
        using var reshapedLogits = logits.reshape(output.size(0), output.size(1), logits.size(-1));

        return reshapedLogits; // Caller takes ownership
    }

    public override Tensor forward(Tensor input)
    {
        return forward(input, null);
    }
}