# GTAD: A Graph-based Approach for Cell Spatial Composition Inference from Integrated scRNA-seq and ST-seq Data

## Requirements

### python

 - tensorflow=2.12.0
 - scanpy=1.9.3
 - numpy =1.23.5
 - pandas=2.0.2
 - scikit-learn=1.0.2
 - scipy 1.11.1
### R
 - R=4.2.0 
 - Seurat=4.3.0 
 - DropletUtils=1.18.1

## Run the model
This model requires you to create a 'data' folder in the current directory to store the data for both scRNA-seq and ST. 

 - For scRNA-seq, you will need 'scRNA_data.csv' and 'scRNA_meta.csv',
   which respectively represent the gene expression matrix of its cells
   and metadata. The gene expression matrix should be in the 'gene×cell'
   format.  
 - For the ST data, you will need 'ST_data.csv', representing    the
   gene expression matrix of spots, in the 'gene×spot' format.

```
python geneFilter.py
Rscript makePusuo.R
python getGraph.py
python runModel.py
```
Then you will get your results.
