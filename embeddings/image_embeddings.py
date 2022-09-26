import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
import os

import argparse
import pandas as pd
import timm
import warnings

"""Largely inspired by https://github.com/kozodoi/website/blob/master/_notebooks/2021-05-27-extracting-features.ipynb"""

class ImageData(Dataset):
    """From https://github.com/kozodoi/website/blob/master/_notebooks/2021-05-27-extracting-features.ipynb"""

    # init
    def __init__(self,
                 data,
                 directory,
                 transform):
        self.data = data
        self.directory = directory
        self.transform = transform

    # length
    def __len__(self):
        return len(self.data)

    # get item
    def __getitem__(self, idx):
        # import
        image = cv2.imread(os.path.join(self.directory, self.data.iloc[idx]['image_id']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # augmentations
        image = self.transform(image=image)['image']

        return image


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_splits", type=int, default=10)
    parser.add_argument(
        "--csv", default="../analysis/materials_w-sent-identifier.csv")
    parser.add_argument("--embedding_model", default="resnet50") # distilgpt2
    parser.add_argument("--embedding_model_layer", default='output')
    parser.add_argument("--distinctiveness", action="store_true")
    args = parser.parse_args()

    if args.embedding_model == "resnet50":

        ##### DATA LOADER

        # import data
        # Go into '../sent_imgs/' folder and write a csv with the image names in that folder
        os.chdir("../sent_imgs/")
        if not os.path.exists("image_names.csv"):
            os.system("ls > image_names.csv")

        # Read the csv with the image names
        data = pd.read_csv("../sent_imgs/image_names.csv", header=None)

        # Only get the ones that don't start with filler
        data = data[data[0].str.startswith("filler") == False].reset_index(drop=True)

        # Rename the column
        data.columns = ["image_id"]

        # Get the first part (string) as a condition name and the second part (number) as a sentence number
        data["condition"] = data["image_id"].str.split(".", expand=True)[0]
        data["item_id"] = data["image_id"].str.split(".", expand=True)[1]

        # Remove row with image_names cond
        data = data[data["condition"] != "image_names"].reset_index(drop=True)

        # data.value_counts("condition")

        # augmentations
        transforms = A.Compose([A.Resize(height=224, width=224),
                                A.Normalize(),
                                ToTensorV2()])

        # dataset
        data_set = ImageData(data=data,
                             directory='../sent_imgs/',
                             transform=transforms)

        # dataloader
        data_loader = DataLoader(data_set,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=2)

        ##### MODEL
        device = torch.device('cpu')

        model = timm.create_model(model_name='resnet50', pretrained=True)
        model.eval()
        model.to(device)

        ##### FEATURE EXTRACTION LOOP

        # placeholders
        PREDS = []
        FEATS = []

        # placeholder for batch features
        features = {}

        # loop through batches
        for idx, inputs in enumerate(data_loader):

            # move to device
            inputs = inputs.to(device)

            # forward pass [with feature extraction]
            if args.embedding_model_layer == 'output':
                with torch.no_grad():
                    preds = model(inputs)
            else:
                # currently not in use!
                warnings.warn("Currently not in use!")
                with torch.no_grad():
                    preds, feats = model(inputs, return_features=True)
                    features[idx] = feats[args.embedding_model_layer].detach().cpu().numpy()

            # Add to the data dataframe in a list for each row (input)
            PREDS.append(preds.detach().cpu().numpy().squeeze()) # squeeze to remove the batch dimension
            # FEATS.append(features['feats'].cpu().numpy())


    else:
        raise ValueError(f"Unrecognized embedding model: {args.embedding_model}")

    # Add the predictions to the data dataframe
    data[f"{args.embedding_model}_{args.embedding_model_layer}"] = PREDS


    # Save the dataframe
    data.to_csv(f"../model-actv/{args.embedding_model}/{args.embedding_model_layer}.csv", index=False)


