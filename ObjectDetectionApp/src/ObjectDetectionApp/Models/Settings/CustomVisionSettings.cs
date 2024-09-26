using Microsoft.Extensions.Configuration;

namespace ObjectDetectionApp.Models.Settings
{
    public class CustomVisionSettings
    {
        public string TrainingEndpoint { get; set; }
        public string TrainingKey { get; set; }
        public string PredictionEndpoint { get; set; }
        public string PredictionKey { get; set; }
        public string PredictionResourceId { get; set; }
    }

    public class CustomVision
    {
        private readonly IConfiguration _configuration;

        public CustomVision(IConfiguration configuration)
        {
            _configuration = configuration;
        }

        public CustomVisionSettings GetSettings()
        {
            var customVision = new CustomVisionSettings();
            _configuration.GetSection("CustomVisionSettings").Bind(customVision);
            return customVision;
        }
    }
}
