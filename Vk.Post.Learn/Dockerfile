FROM mcr.microsoft.com/dotnet/runtime:6.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:6.0 AS build
WORKDIR /src
COPY ["Vk.Post.Learn/Vk.Post.Learn.csproj", "Vk.Post.Learn/"]
RUN dotnet restore "Vk.Post.Learn/Vk.Post.Learn.csproj"
COPY . .
WORKDIR "/src/Vk.Post.Learn"
RUN dotnet build "Vk.Post.Learn.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "Vk.Post.Learn.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "Vk.Post.Learn.dll"]
