import os
import warnings
from tensorflow.keras.models import load_model
from Functions import main  # Assuming 'Functions' is a local module
from tensorflow.keras.models import load_model
import sys

warnings.filterwarnings("ignore")

# %% Usage Example
dataset_types = ["Residential", "Manufacturing facility", "Office building", "Retail store", "Medical clinic"]
model_types = ['LSTM', 'GRU', 'BLSTM', 'BGRU', 'CNN', 'TCN', 'DCNN', 'WaveNet', 'TFT', 'TST']

dataset_type = dataset_types[0]

# Load your data
mydata = main.load_and_preprocess_data(dataset_type=dataset_type)
# print(mydata.X_train.shape) #(20391, 48, 10)
# print(mydata.X_val.shape) #(2548, 48, 10)
# print(mydata.X_test.shape) #(2548, 48, 10)
#
# print(mydata.y_train.shape) #(20391, 24, 1)
# print(mydata.y_val.shape) #(2548, 24, 1)
# print(mydata.y_test.shape) #(2548, 24, 1)


model_dir = f"./Results/Models/{mydata.data_type}"

# for model_type in model_types:
model_type = model_types[0]
# model_type = model_types[9]
best_model = load_model(f"{model_dir}/{model_type}.keras")

# Define prediction function
def model_predict(X):
    # Reshape if needed (handle both 2D and 3D inputs)
    if X.ndim == 2:
        X = X.reshape(-1, mydata.X.shape[1], mydata.X.shape[2])
    return best_model.predict(X, verbose=0)[:, 0]  # Return first horizon predictions
#
# if os.path.exists(f"./Results/DiffusionExplainer/{mydata.data_type}/{model_type}_DiffusionExplainer.pt"):
#     print(f"Explainer for {model_type} exist. Skipped training ...")
#
# # Initialize explainer
# explainer = main.VITExplainer(
#     n_epochs=150,
#     batch_size=256,
# ).initialize(
#     X_train=mydata.X[:mydata.X_train.shape[0] + mydata.X_val.shape[0], :, :],
#     model_predict_func=model_predict,
#     feature_names=mydata.feature_names,
# )
#
# explainer.save(f"./Results/DiffusionExplainer/{dataset_type}/", filename=f"{model_type}_DiffusionExplainer")
#
# loaded_explainer = main.DiffusionExplainer.load(f"./Results/DiffusionExplainer/{dataset_type}/",filename=f"{model_type}_DiffusionExplainer")
#
# explainer.explain(mydata.X_train[0:1,:,:])


