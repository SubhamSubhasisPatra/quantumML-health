
library(readr)
library(viridis)
library(dplyr)
library(ggplot2)
library(ggpubr)
library(stringr)
library(viridis)
library(data.table)




#============================
#         xor lite
#============================

##----- average search

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

###----- epoch error

xornobias = read_csv('results/epoch_error_xorlite_nobias.csv')
xornobias <- melt(setDT(xornobias), id.vars = c("X1","epoch"), variable.name = "model")
p1 = ggplot(xornobias, aes(x=epoch, y=value, group=model, color=model)) +
  geom_line(size=1) +
  scale_color_viridis(discrete = TRUE) +
  labs(x='Epoch', y="Epoch Error", title='Without Bias')+
  theme_minimal()+
  theme(legend.title = element_blank(), 
        legend.text = element_text(size=11))+
  guides(color = guide_legend(override.aes = list(size = 4) ) )

p1

xorbias = read_csv('results/epoch_error_xorlite_bias.csv')
xorbias <- melt(setDT(xorbias), id.vars = c("X1","epoch"), variable.name = "model")
p2 = ggplot(xorbias, aes(x=epoch, y=value, group=model, color=model)) +
  geom_line(size=1) +
  scale_color_viridis(discrete = TRUE) +
  labs(x='Epoch', y="Epoch Error", title='With Bias')+
  theme_minimal() +
  theme(legend.title = element_blank(), 
        legend.text = element_text(size=11))+
  guides(color = guide_legend(override.aes = list(size = 4)) )

p2

ggarrange(p1, p2, ncol = 1, nrow = 2, common.legend = T, legend='bottom')
ggsave('results/xor-epoch.png', height = 5, width = 7, units = 'in')  

#==============
# xor real
#=============

data = read_csv('experiment_real_xor.csv')

data$phase_strategy = paste0("phase-encoding\n", data$phase_strategy)

data$phase_strategy[data$model == 'HSGS'] = 'HSGS'

aggregate(data$avg_error, by=list(data$phase_strategy), mean)

ggplot(data, aes(x=phase_strategy, y=avg_error, fill=phase_strategy)) + 
  geom_boxplot()+
  scale_fill_viridis(name = '', discrete = T)+
  labs(x='Quantum Neuron Model', y="Error")+
  theme_minimal()+ 
  theme(legend.position = "none") 

ggsave('results/xor-real-models.png', height = 4, width = 7, units = 'in')  
