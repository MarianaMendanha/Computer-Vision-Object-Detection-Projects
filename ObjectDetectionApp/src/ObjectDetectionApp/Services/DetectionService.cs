using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Prediction;
using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Prediction.Models;
using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Training;
using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Training.Models;
using Microsoft.Extensions.Configuration;
using System.IO;

using ObjectDetectionApp.Models;
using ObjectDetectionApp.Models.Settings;
using ObjectDetectionApp.Tools;

public class Program
{
    public static void GetProject()
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

        string projectId = "14397381-8c2c-4ac0-a342-eebf95254304";
        Project project = modelTrainer.GetProject(projectId);
        Console.WriteLine($"Project Name: {project.Name}");
        
    }
}