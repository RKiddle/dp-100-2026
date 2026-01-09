# import required libraries
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
import argparse

# Enter details of your Azure Machine Learning workspace
subscription_id = "<SUBSCRIPTION_ID>"
resource_group = "<RESOURCE_GROUP>"
workspace_name = "<WORKSPACE_NAME>"

def main():
    # Connect to the workspace
    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id,
        resource_group,
        workspace_name
    )

    # Define the Data asset
    # We are registering a local file as a versioned data asset
    my_data = Data(
        path="./data/diabetes.csv",
        type=AssetTypes.URI_FILE,
        description="Diabetes dataset for DP-100 training",
        name="diabetes-data",
        version="1"
    )

    # Create the data asset in the workspace
    print(f"Registering data asset: {my_data.name}...")
    ml_client.data.create_or_update(my_data)
    print(f"Data asset {my_data.name} registered successfully.")

if __name__ == "__main__":
    main()
