using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace BasicGenerativeAI.System;

public class ConversationHistory
{
    private readonly List<(string speaker, string text)> _history;
    private readonly int _maxTurns; // Opcional: Limitar o histórico para não ficar muito longo

    // Construtor
    public ConversationHistory(int maxTurns = 20) // Exemplo: Manter os últimos 20 turnos
    {
        _history = new List<(string speaker, string text)>();
        _maxTurns = maxTurns;
    }

    // Adiciona um turno (fala) ao histórico
    public void AddTurn(string speaker, string text)
    {
        _history.Add((speaker, text));

        // Remover turnos antigos se o histórico exceder o limite
        while (_history.Count > _maxTurns)
        {
            _history.RemoveAt(0);
        }
    }

    // Limpa todo o histórico
    public void Clear()
    {
        _history.Clear();
    }

    // Obtém o histórico formatado como uma única string para servir de input ao modelo.
    // Formato estilo:
    // User: Primeira fala do usuário
    // AI: Primeira resposta da AI
    // User: Segunda fala do usuário
    // AI:
    public string GetFormattedHistory()
    {
        var sb = new StringBuilder();
        foreach (var turn in _history)
        {
            sb.AppendLine($"{turn.speaker}: {turn.text}");
        }
        // Adiciona o prompt para a resposta da AI
        sb.Append("AI:"); // AI: [Espaço para a AI começar a gerar]

        return sb.ToString();
    }

    // Retorna o histórico completo (não formatado)
    public List<(string speaker, string text)> GetHistory()
    {
        return new List<(string speaker, string text)>(_history); // Retorna uma cópia
    }
}