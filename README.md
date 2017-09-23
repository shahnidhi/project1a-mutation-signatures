# Project 1a: Mutation Signatures

For this project, you will implement and run the NMF mutation signature decomposition described in [Alexandrov, et al. (Cell Reports, 2013)](https://www.nature.com/nature/journal/v500/n7463/full/nature12477.html) on the data from [Alexandrov, et al. (Nature, 2013)](http://www.cell.com/cell-reports/abstract/S2211-1247(12)00433-0).

### Data

The input data for your algorithm is a mutation count matrix. The mutation count matrices are stored in a tab-separated text file, where each line lists the number of mutations of a given category for a single patient. The patient name will be stored in the first column, and the category names in the first row.

#### Example data

You can find a small example dataset for your project in [data/examples](https://github.com/cmsc828p-f17/project1a-mutation-signatures/blob/master/data/examples). The examples directory also includes the signatures used to generate the data, which you can use to sanity check your results.

#### Real data

You will need to download real data for your project and process it into the same format as the example data. You can find the Pan-Cancer mutation counts originally used by Alexandrov, et al. (Nature, 2013) at [ftp://ftp.sanger.ac.uk/pub/cancer/AlexandrovEtAl](ftp://ftp.sanger.ac.uk/pub/cancer/AlexandrovEtAl).
