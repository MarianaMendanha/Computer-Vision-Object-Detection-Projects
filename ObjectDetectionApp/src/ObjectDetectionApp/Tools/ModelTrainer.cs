using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Prediction;
using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Prediction.Models;
using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Training;
using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Training.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Microsoft.Extensions.Configuration;
using System.IO;

using ObjectDetectionApp.Models;
using ObjectDetectionApp.Models.Settings;

namespace ObjectDetectionApp.Tools
{
    public class ModelTrainer
    {
        private CustomVision _customVision;
        private readonly CustomVisionSettings _customVisionSettings;
        private CustomVisionTrainingClient _trainingApi;
        private CustomVisionPredictionClient _predictionApi;

        public ModelTrainer(CustomVision customVision)
        {
            _customVision = customVision;
            _customVisionSettings = _customVision.GetSettings();
            _trainingApi = AuthenticateTraining(_customVisionSettings.TrainingEndpoint, _customVisionSettings.TrainingKey);
            _predictionApi = AuthenticatePrediction(_customVisionSettings.PredictionEndpoint, _customVisionSettings.PredictionKey);
        }

        private static CustomVisionTrainingClient AuthenticateTraining(string endpoint, string trainingKey)
        {
            // Create the Api, passing in the training key
            CustomVisionTrainingClient trainingApi = new CustomVisionTrainingClient(new Microsoft.Azure.CognitiveServices.Vision.CustomVision.Training.ApiKeyServiceClientCredentials(trainingKey))
            {
                Endpoint = endpoint
            };
            return trainingApi;
        }

        private static CustomVisionPredictionClient AuthenticatePrediction(string endpoint, string predictionKey)
        {
            // Create a prediction endpoint, passing in the obtained prediction key
            CustomVisionPredictionClient predictionApi = new CustomVisionPredictionClient(new Microsoft.Azure.CognitiveServices.Vision.CustomVision.Prediction.ApiKeyServiceClientCredentials(predictionKey))
            {
                Endpoint = endpoint
            };
            return predictionApi;
        }
        
        public Project CreateProject(string projectName)
        {
            // Find the object detection domain
            var domains = _trainingApi.GetDomains();
            var objDetectionDomain = domains.FirstOrDefault(d => d.Type == "ObjectDetection");

            // Create a new project
            Console.WriteLine("Creating new project:");
            var project = _trainingApi.CreateProject(projectName, null, objDetectionDomain.Id);

            return project;
        }

        public Project GetProject(string projectId)
        {
            Guid projectIdGuid = new Guid(projectId);
            var project = _trainingApi.GetProject(projectIdGuid);
            return project;
        }

        public Dictionary<string, Tag> AddTags(Project project, List<string> tagNames)
        {
            var newTags = new Dictionary<string, Tag>();
            foreach (var tagName in tagNames)
            {
                var newTag = _trainingApi.CreateTag(project.Id, tagName); 
                newTags.Add(newTag.Name, newTag);
            }

            return newTags;
        }

        public IList<Tag> GetTags(string projectId)
        {
            Guid projectIdGuid = new Guid(projectId);
            // Obtenha as tags do projeto
            var tags = _trainingApi.GetTags(projectIdGuid);

            return tags;
        }
        
        public bool UploadImageWithTags(Project project, string imageName, IList<Tag> tags, Dictionary<string, double[]> tagNameAndRegion)
        {   
            var imageFile = Path.Combine("..", "data", "TrainImages", imageName);
            try
            {
                // double[] region: Left, Top, Width, Height
                var tagRegions = new Dictionary<Tag, double[]>();

                foreach (var tag in tags)
                {
                    if (tagNameAndRegion.TryGetValue(tag.Name, out var region))
                    {
                        tagRegions[tag] = region;
                    }
                }

                // Cria uma lista de regiões associadas a cada tag
                var regions = new List<Region>();
                foreach (var tagRegion in tagRegions)
                {
                    Tag tag = tagRegion.Key;
                    double[] region = tagRegion.Value;
                    regions.Add(new Region(tag.Id, region[0], region[1], region[2], region[3]));
                }

                // Cria uma entrada de arquivo de imagem para a imagem com várias tags
                var imageFileEntry = new ImageFileCreateEntry(imageFile, File.ReadAllBytes(imageFile), null, regions);

                // Adiciona a imagem ao projeto
                _trainingApi.CreateImagesFromFiles(project.Id, new ImageFileCreateBatch(new List<ImageFileCreateEntry> { imageFileEntry }));

                return true;
            }
            catch (Exception)
            {
                // Se ocorrer um erro durante o upload da imagem, retorna false
                return false;
            }
        }

        public void TrainProject(Project project, string modelName)
        {

            // Now there are images with tags start training the project
            Console.WriteLine("\tTraining");
            var iteration = _trainingApi.TrainProject(project.Id);

            // The returned iteration will be in progress, and can be queried periodically to see when it has completed
            while (iteration.Status == "Training")
            {
                Thread.Sleep(1000);

                // Re-query the iteration to get its updated status
                iteration = _trainingApi.GetIteration(project.Id, iteration.Id);
            }

            //PublishIteration(trainingApi, project, modelName, "");
        }

        public void PublishIteration(Project project, string modelName, string iterationId)
        {
            Guid iterationIdGuid = new Guid(iterationId);
            // The iteration is now trained. Publish it to the prediction end point.
            _trainingApi.PublishIteration(project.Id, iterationIdGuid, modelName, _customVisionSettings.PredictionResourceId);
            Console.WriteLine("Done!\n");
        }

        public IList<Iteration> ListIterations(Project project)
        {
            Console.WriteLine("Listing iterations:");
            var iterations = _trainingApi.GetIterations(project.Id);

            foreach (var iteration in iterations)
            {
                Console.WriteLine($"\tName: {iteration.Name}, Status: {iteration.Status}, Id: {iteration.Id}");
            }
            return iterations;
        }

        public IList<PredictionModel> TestIteration(Project project, string imageName, string publishedModelName)
        {
            // Make a prediction
            Console.WriteLine("Making a prediction:");
            var imageFile = Path.Combine("..", "..", "data", "testing_images", imageName);
            Console.WriteLine(imageFile);
            using (var stream = File.OpenRead(imageFile))
            {
                var result = _predictionApi.DetectImage(project.Id, publishedModelName, stream);
                return result.Predictions;
            }
        }

        public IList<Image> GetTaggedImages(Project project)
        {
            // Obter todas as imagens marcadas no projeto
            var taggedImages = _trainingApi.GetTaggedImages(project.Id);

            return taggedImages;
        }

        public async Task TransferImages(Project sourceProject, Project targetProject)
        {
            // Obtem todas as imagens marcadas no projeto de origem
            var taggedImages = GetTaggedImages(sourceProject);

            // Cria uma lista para armazenar as entradas de arquivo de imagem
            var imageFileEntries = new List<ImageFileCreateEntry>();

            // Cria um dicionário para armazenar as tags do projeto de destino
            var targetTags = new Dictionary<string, Guid>();

            // Obtém todas as tags do projeto de destino
            var existingTags = await _trainingApi.GetTagsAsync(targetProject.Id);

            // Adiciona as tags existentes ao dicionário
            foreach (var tag in existingTags)
            {
                targetTags[tag.Name] = tag.Id;
            }

            // Itera sobre cada imagem marcada
            foreach (var taggedImage in taggedImages)
            {
                Console.WriteLine($"Image ID: {taggedImage.Id}  ");
                // Cria uma lista para armazenar as regiões
                var regions = new List<Region>();
                foreach (var region in taggedImage.Regions)
                {
                    Console.WriteLine($"Tag Name: {region.TagName}");

                    // Verifica se a tag já existe no projeto de destino
                    if (!targetTags.ContainsKey(region.TagName))
                    {
                        // Se não existir, cria a tag no projeto de destino
                        var createdTag = await _trainingApi.CreateTagAsync(targetProject.Id, region.TagName);
                        targetTags[region.TagName] = createdTag.Id;
                    }

                    regions.Add(new Region(targetTags[region.TagName], region.Left, region.Top, region.Width, region.Height));
                }

                using (var client = new HttpClient())
                {
                    var response = await client.GetAsync(taggedImage.OriginalImageUri);

                    using (var ms = new MemoryStream(await response.Content.ReadAsByteArrayAsync()))
                    {
                        var imageFileEntry = new ImageFileCreateEntry(taggedImage.OriginalImageUri, ms.ToArray(), null, regions);
                        // Adiciona a entrada de arquivo de imagem à lista de entradas de arquivo de imagem
                        imageFileEntries.Add(imageFileEntry);
                    }
                }
            }

            // Faz o upload das imagens para o novo projeto
            await _trainingApi.CreateImagesFromFilesAsync(targetProject.Id, new ImageFileCreateBatch(imageFileEntries));
        }
        
    }
}