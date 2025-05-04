using TorchSharp;

namespace BasicGenerativeAI.Core;

public abstract class BaseGenerativeModel : IDisposable
{
    // Propriedade abstrata para o tamanho do vocabulário
    public abstract int VocabularySize { get; }

    // Método abstrato para carregar o modelo (pesos) de um arquivo
    public abstract void Load(string filePath);

    // Método abstrato para salvar o modelo (pesos) em um arquivo
    public abstract void Save(string filePath);

    // Método abstrato para gerar uma sequência de tokens a partir de um input inicial
    // inputTokens: Tensor contendo os tokens de entrada (ex: histórico da conversa tokenizado)
    // maxTokens: Número máximo de tokens a gerar na resposta
    // temperature: Controla a aleatoriedade da geração (valores menores -> mais determinístico)
    public abstract torch.Tensor Generate(torch.Tensor inputTokens, int maxTokens, float temperature = 1.0f);

    // Implementação de IDisposable para gerenciar recursos não gerenciados (Tensor, Module, etc.)
    private bool disposedValue;

    protected virtual void Dispose(bool disposing)
    {
        if (!disposedValue)
        {
            if (disposing)
            {
                // Dispor recursos gerenciados
            }

            // Dispor recursos não gerenciados (campos TorchSharp devem ser dispostos aqui)
            // Classes derivadas devem sobrescrever e chamar base.Dispose(true)
            disposedValue = true;
        }
    }

    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }
}