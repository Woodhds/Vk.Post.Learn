using Microsoft.ML.Data;

namespace Vk.Post.Learn
{
    public class VkMessagePredict
    {
        [ColumnName("PredictedLabel")]
        public string Category;

        public float[] Score;
    }
}