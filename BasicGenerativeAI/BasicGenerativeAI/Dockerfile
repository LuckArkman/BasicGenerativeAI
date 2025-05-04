FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS base
WORKDIR /app
EXPOSE 80
EXPOSE 443

FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY ["BasicGenerativeAI/BasicGenerativeAI.csproj", "BasicGenerativeAI/"]
RUN dotnet restore "BasicGenerativeAI/BasicGenerativeAI.csproj"
COPY . .
WORKDIR "/src/BasicGenerativeAI"
RUN dotnet build "BasicGenerativeAI.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "BasicGenerativeAI.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "BasicGenerativeAI.dll"]
