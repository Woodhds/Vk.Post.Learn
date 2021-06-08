using System;
using Microsoft.ML;

namespace Vk.Post.Learn
{
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length < 1)
            {
                Console.WriteLine("Invalid path");
                return;
            }
            
            var messagePredictModelService = new MessagePredictModelService(args[0]);
            var data = messagePredictModelService.Load();
            var mlContext = new MLContext();
            var trainingDataView = mlContext.Data.LoadFromTextFile<VkMessageML>(args[1], ',');

            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(VkMessageML.Category))
                .Append(mlContext.Transforms.Text.NormalizeText("NormalizedText", nameof(VkMessageML.Text)))
                .Append(mlContext.Transforms.Text.NormalizeText("NormalizedOwnerName", nameof(VkMessageML.OwnerName)))
                .Append(mlContext.Transforms.Text.FeaturizeText("FeaturedOwnerName", "NormalizedOwnerName"))
                .Append(mlContext.Transforms.Text.FeaturizeText("FeaturedText", "NormalizedText"))
                .Append(mlContext.Transforms.Concatenate("Features", "FeaturedText", "FeaturedOwnerName"))
                .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy())
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"))
                .AppendCacheCheckpoint(mlContext);

            var transformedNewData = data.Transformer.Transform(trainingDataView);
            mlContext.MulticlassClassification.CrossValidate(transformedNewData, pipeline);
            var trainedModel = pipeline.Fit(transformedNewData);

            messagePredictModelService.Save(mlContext, trainedModel, transformedNewData);
        }
    }
}