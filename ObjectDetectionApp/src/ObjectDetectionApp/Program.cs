using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Prediction;
using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Prediction.Models;
using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Training;
using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Training.Models;
using Microsoft.Extensions.Configuration;
using System.IO;

using ObjectDetectionApp.Models;
using ObjectDetectionApp.Models.Settings;
using ObjectDetectionApp.Tools;

public class DetectionService
{
    public static void Main(string[] args)
    {
        var builder = new ConfigurationBuilder()
        .SetBasePath(Directory.GetCurrentDirectory())
        .AddJsonFile(Path.Combine(Directory.GetParent(Directory.GetCurrentDirectory()).FullName, "config", "appsettings.json"), optional: true, reloadOnChange: true);
        IConfigurationRoot configuration = builder.Build();

        var customVision = new CustomVision(configuration);
        var customVisionSettings = new CustomVisionSettings();
        customVisionSettings = customVision.GetSettings();
        Console.WriteLine(customVisionSettings.TrainingEndpoint);

        var modelTrainer = new ModelTrainer(customVision);
        
        string projectId = "14397381-8c2c-4ac0-a342-eebf95254304";  // Custom Vision AI Project ID -> Esse é só um exemplo, não irá funcionar
        Project project = modelTrainer.GetProject(projectId);
        Console.WriteLine($"Project Name: {project.Name}");

        // detection service
        // var result = modelTrainer.TestIteration(project, "53.jpg", "Iteration 1");
        // var orderedResult = result.OrderByDescending(c => c.Probability).ToList();
        // foreach (var c in orderedResult)
        // {
        //     // Imprime com um formato padronizado
        //     Console.WriteLine($"\t{c.TagName.PadRight(15)}: {c.Probability:P1} [ {c.BoundingBox.Left}, {c.BoundingBox.Top}, {c.BoundingBox.Width}, {c.BoundingBox.Height} ]");
        // }

        // pegar dataset em formato kitti
        var taggedImages = modelTrainer.GetTaggedImages(project);

        foreach (var image in taggedImages)
        {
            Console.WriteLine(image.Id + ": ");
            foreach (var region in image.Regions)
            {
                Console.WriteLine("\t\t\t\t\t" + region.TagName);
            }
        }
        
    }
}
