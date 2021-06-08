using Microsoft.ML;

namespace Vk.Post.Learn
{
    public class MLContextModel
    {
        public ITransformer Transformer { get; set; }
        public MLContext Context { get; set; }
        public DataViewSchema Schema { get; set; }
    }
}