using Microsoft.Extensions.Configuration;

namespace ObjectDetectionApp.Models.Settings
{
    public class ClientSettings
    {
        public string TenantId { get; set; }
        public string SubscriptionId { get; set; }
        public string ResourceGroupName { get; set; }
        public string WorkspaceName { get; set; }
    }

    public class Client
    {
        private readonly IConfiguration _configuration;

        public Client(IConfiguration configuration)
        {
            _configuration = configuration;
        }

        public ClientSettings GetSettings()
        {
            var clientSettings = new ClientSettings();
            _configuration.GetSection("ClientSettings").Bind(clientSettings);
            return clientSettings;
            //string variavelURL = _configuration["VariavelURL"];
        }
    }
}
