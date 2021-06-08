using Microsoft.ML;

namespace Vk.Post.Learn
{
    public class MessagePredictModelService
    {
        private readonly string _path;
        
        public MessagePredictModelService(string path)
        {
            _path = path;
        }
        
        public MLContextModel Load()
        {
            var context = new MLContext();
            var iTransformer = context.Model.Load(_path, out var schema);
            return new MLContextModel
            {
                Context = context,
                Transformer = iTransformer,
                Schema = schema
            };
        }

        public void Save(MLContext context, ITransformer transformer, IDataView dataView)
        {
            context.Model.Save(transformer, dataView.Schema, _path);
        }
    }
}