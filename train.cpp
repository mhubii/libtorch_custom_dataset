#include <torch/torch.h>

#include "custom_dataset.h"
#include "model.h"

int main()
{
    // Load the model.
    ConvNet model(3/*channel*/, 64/*height*/, 64/*width*/);

    // Generate your data set. At this point you can add transforms to you data set, e.g. stack your
    // batches into a single tensor.
    std::string file_names_csv = "../file_names.csv";
    auto data_set = CustomDataset(file_names_csv).map(torch::data::transforms::Stack<>());

    // Generate a data loader.
    int64_t batch_size = 32;
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        data_set,
        batch_size);

    // Chose and optimizer.
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));

    // Train the network.
    int64_t n_epochs = 10;
    int64_t log_interval = 10;
    int dataset_size = data_set.size().value();

    // Record best loss.
    float best_mse = std::numeric_limits<float>::max();

    for (int epoch = 1; epoch <= n_epochs; epoch++) {

        // Track loss.
        size_t batch_idx = 0;
        float mse = 0.; // mean squared error
        int count = 0;

        for (auto& batch : *data_loader) {
            auto imgs = batch.data;
            auto labels = batch.target.squeeze();

            imgs = imgs.to(torch::kF32);
            labels = labels.to(torch::kInt64);

            optimizer.zero_grad();
            auto output = model(imgs);
            auto loss = torch::nll_loss(output, labels);

            loss.backward();
            optimizer.step();

            mse += loss.template item<float>();

            batch_idx++;
            if (batch_idx % log_interval == 0) 
            {
                std::printf(
                "\rTrain Epoch: %d/%ld [%5ld/%5d] Loss: %.4f",
                epoch,
                n_epochs,
                batch_idx * batch.data.size(0), 
                dataset_size,
                loss.template item<float>());
            }

            count++;
        }

        mse /= (float)count;
        printf(" Mean squared error: %f\n", mse);   

        if (mse < best_mse)
        {
            torch::save(model, "../best_model.pt");
            best_mse = mse;
        }
    }  

    return 0;
}