from tslearn.datasets import UCR_UEA_datasets

uv_datasets = UCR_UEA_datasets().list_univariate_datasets()
print(uv_datasets)
mv_datasets = UCR_UEA_datasets().list_multivariate_datasets()
print(mv_datasets)
datasets = UCR_UEA_datasets().list_datasets()
print(datasets)

print('It works')
