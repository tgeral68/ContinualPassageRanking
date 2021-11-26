# Dataset


## Loading datasets

For loading continual dataset you should use the continual_dataset package. At the moment we only load MsMarco corpus based on the **ir_datasets** library. You can either laod the raw ms-passage ranking corpus or scenarios based on pre-computed script (see ressources folder for dataset generated in the associated paper).

To load a dataset based on a split:

```python
topics_folder_path = "ressources/V1.0/MsMarco-continual-medium"
continual_dataset = continual_dataset.MSMarcoRankingDataset(topics_folder_path)
```
**!!! Notice that first call to the dataset class will download the MsMarco V1 corpus !!!**, you also will need enough ram to load it into memory ~20Gb.
Once you get the dataset is loaded you can acess element using the `dataset[i]` syntax to get elements and change tasks. For instance:

``` python


```