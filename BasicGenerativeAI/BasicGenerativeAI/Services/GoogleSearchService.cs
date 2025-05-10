using System.Text.Json; // Ou System.Text.Json.Nodes para .NET 6+
using System.Net.Http;
using System.Threading.Tasks;
using System.Collections.Generic;

namespace BasicGenerativeAI.Services;

public class GoogleSearchService : IDisposable
{
        private readonly HttpClient _httpClient;
        private readonly string _apiKey; // Chave da API do Google Cloud
        private readonly string _cx;     // ID do Custom Search Engine

        private const string SearchApiUrl = "https://www.googleapis.com/customsearch/v1";

        // Construtor
        public GoogleSearchService(string apiKey, string cx)
        {
            if (string.IsNullOrEmpty(apiKey) || string.IsNullOrEmpty(cx))
            {
                throw new ArgumentException("API Key and CX must be provided for Google Search.");
            }

            _apiKey = apiKey;
            _cx = cx;
            _httpClient = new HttpClient();
        }

        // Realiza uma busca e retorna uma lista de resultados formatados
        public async Task<List<string>> SearchAsync(string query, int numResults = 3)
        {
            var results = new List<string>();

            if (string.IsNullOrWhiteSpace(query))
            {
                results.Add("Busca vazia. Nenhum resultado.");
                return results;
            }

            try
            {
                // Constrói a URL da requisição
                var url = $"{SearchApiUrl}?key={Uri.EscapeDataString(_apiKey)}&cx={Uri.EscapeDataString(_cx)}&q={Uri.EscapeDataString(query)}&num={numResults}";

                Console.WriteLine($"Realizando busca: {query}");
                var response = await _httpClient.GetAsync(url);
                response.EnsureSuccessStatusCode(); // Lança exceção para códigos de erro HTTP

                var jsonString = await response.Content.ReadAsStringAsync();

                // Processa a resposta JSON
                using var doc = JsonDocument.Parse(jsonString);
                var root = doc.RootElement;

                if (root.TryGetProperty("items", out var itemsElement) && itemsElement.ValueKind == JsonValueKind.Array)
                {
                    foreach (var item in itemsElement.EnumerateArray())
                    {
                        if (item.TryGetProperty("title", out var titleElement) &&
                            item.TryGetProperty("snippet", out var snippetElement) &&
                            item.TryGetProperty("link", out var linkElement))
                        {
                             var title = titleElement.GetString();
                             var snippet = snippetElement.GetString();
                             var link = linkElement.GetString();
                            if (!string.IsNullOrEmpty(title) && !string.IsNullOrEmpty(snippet))
                            {
                                // Formata o resultado para ser útil para o modelo
                                results.Add($"Título: {title}\nLink: {link}\nSnippet: {snippet}\n");
                            }
                        }
                    }
                }

                if (results.Count == 0)
                {
                    results.Add("Nenhum resultado encontrado para a busca.");
                }

            }
            catch (HttpRequestException httpEx)
            {
                results.Add($"Erro HTTP ao realizar busca: {httpEx.Message}");
                 Console.WriteLine($"Erro HTTP: {httpEx}");
            }
             catch (JsonException jsonEx)
            {
                results.Add($"Erro ao processar resposta da busca: {jsonEx.Message}");
                 Console.WriteLine($"Erro JSON: {jsonEx}");
            }
            catch (Exception ex)
            {
                results.Add($"Ocorreu um erro inesperado durante a busca: {ex.Message}");
                Console.WriteLine($"Erro inesperado: {ex}");
            }

            return results;
        }

        // Dispor o HttpClient se necessário (embora Singleton HttpClient seja prática comum,
        // para este exemplo simples, podemos manter a instância aqui).
        public void Dispose()
        {
            Dispose();
            GC.SuppressFinalize(this);
        }
    }