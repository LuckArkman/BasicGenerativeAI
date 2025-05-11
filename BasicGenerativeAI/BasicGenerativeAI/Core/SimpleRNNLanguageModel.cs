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
    private readonly Device _device;
    private readonly int _numLayers;
    private readonly int _hiddenDim;

    public SimpleRNNLanguageModel(long vocabSize, Device device, int embeddingDim, int hiddenDim, int rnnLayers) 
        : base("SimpleRNNLanguageModel")
    {
        _device = device ?? throw new ArgumentNullException(nameof(device));
        _numLayers = rnnLayers;
        _hiddenDim = hiddenDim;

        _embedding = Embedding(vocabSize, embeddingDim).to(device);
        _rnn = GRU(inputSize: embeddingDim, hiddenSize: hiddenDim, numLayers: rnnLayers, batchFirst: true).to(device);
        _linear = Linear(hiddenDim, vocabSize).to(device);

        RegisterComponents();

        // Verificação inicial para garantir que a camada GRU tem parâmetros
        var rnnParams = _rnn.parameters();
        if (rnnParams.Count() == 0)
        {
            throw new InvalidOperationException("GRU module has no parameters. Initialization may have failed.");
        }
    }

    public Tensor forward(Tensor input, Tensor? hidden = null)
    {
        if (input.dtype != ScalarType.Int64)
        {
            string shapeStr = $"[{string.Join(", ", input.shape)}]";
            throw new ArgumentException(
                $"Input tensor for embedding must be Int64, but got {input.dtype}. Original shape: {shapeStr}");
        }

        input = input.to(_device);
        Console.WriteLine($"Input shape: [{string.Join(", ", input.shape)}]");

        if (_embedding.weight.Handle == IntPtr.Zero)
        {
            throw new InvalidOperationException("Embedding weights are invalid.");
        }

        using var embedded = _embedding.forward(input);
        if (embedded.Handle == IntPtr.Zero)
        {
            throw new InvalidOperationException("Embedded tensor is invalid after embedding forward.");
        }
        Console.WriteLine($"Embedded shape: [{string.Join(", ", embedded.shape)}]");

        if ((bool)(hidden == null))
        {
            long batchSize = input.size(0);
            hidden = torch.zeros(_numLayers, batchSize, _hiddenDim, device: _device);
        }
        else
        {
            hidden = hidden.to(_device);
        }

        var rnnOutputTuple = _rnn.forward(embedded, hidden);
        using var output = rnnOutputTuple.Item1;
        rnnOutputTuple.Item2?.Dispose();
        if (output.Handle == IntPtr.Zero)
        {
            throw new InvalidOperationException("RNN output tensor is invalid after GRU forward.");
        }
        Console.WriteLine($"RNN output shape: [{string.Join(", ", output.shape)}]");

        using var reshapedOutput = output.reshape(-1, output.size(2));
        using var logits = _linear.forward(reshapedOutput);
        using var reshapedLogits = logits.reshape(output.size(0), output.size(1), logits.size(-1));

        return reshapedLogits;
    }

    public override Tensor forward(Tensor input)
    {
        return forward(input, null);
    }

    public Tensor initHidden(long batchSize)
    {
        return torch.zeros(_numLayers, batchSize, _hiddenDim, device: _device);
    }
}