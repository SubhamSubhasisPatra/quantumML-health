
library(readr)
library(viridis)
library(dplyr)
library(ggplot2)
library(ggpubr)
library(stringr)
library(viridis)




#==============
# xor
#=============

data = read_csv('experiment_xor.csv')

data$phase_strategy = paste0("phase-encoding\n", data$phase_strategy)

data$phase_strategy[data$model == 'HSGS'] = 'HSGS'

aggregate(data$avg_error, by=list(data$phase_strategy), mean)

ggplot(data, aes(x=phase_strategy, y=avg_error, fill=phase_strategy)) + 
  geom_boxplot()+
  scale_fill_viridis(name = '', discrete = T)+
  labs(x='Quantum Neuron Model', y="Error")+
  theme_minimal()+ 
  theme(legend.position = "none") 

ggsave('results/xor-binary-models.png', height = 4, width = 7, units = 'in')  


#==============
# xor real
#=============

data = read_csv('experiment_real_xor.csv')

data$phase_strategy = paste0("phase-encoding\n", data$phase_strategy)

data$phase_strategy[data$model == 'HSGS'] = 'HSGS'


ggplot(data, aes(x=phase_strategy, y=avg_error, fill=phase_strategy)) + 
  geom_boxplot()+
  scale_fill_viridis(name = '', discrete = T)+
  labs(x='Quantum Neuron Model', y="Error")+
  theme_minimal()+ 
  theme(legend.position = "none") 

ggsave('results/xor-real-models.png', height = 4, width = 7, units = 'in')  
