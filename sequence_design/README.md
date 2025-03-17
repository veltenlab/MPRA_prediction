# Sequence design by deep model

Refers to Froemel et al., Figure 6

The code assumes that you have trained the deep model and have a model server running on localhost (see parent folder)

You then also need to train a random forest model using the script `train_random_forest.R`

Then, run, for example:

```
TASK="1,-0.2,-0.2,1,1,1,1" #try to create a sequence with activity 1 in cell state 1,4,5,6,7, and mild repression of -0.2 in the other cell states:
#Mapping of cell state numbers to identifiers:
#State_1M      State_2D      State_3E      State_4M      State_5M      State_6N      State_7M 
#"MegEry"    "Basophil"  "Eosinophil"    "Monocyte" "Monocyte P."  "Neutrophil"    "Immature" 

NSTEPS=10 #number of steps in local search
NSTEPSMCMC=7000 #number of steps in global search
PORT=4567 #the port that your model server listens to
OUTFOLDER=seqdesigns 

mkdir #OUTFOLDER
Rscript evolve_from_commandline_server_final.R $TASK $NSTEPS $NSTEPSMCMC $PORT $OUTFOLDER
```

For the results included in figure 6 we ran this for all tasks from `tasklist.txt`

`Generate_Sequences.R` provides functions to create sequences from a specification (e.g. 3 Gata1 sites, 3 Cebpa sites)
