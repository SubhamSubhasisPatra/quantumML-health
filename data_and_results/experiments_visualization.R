
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

xornobias = read_csv('results/epoch_error_xorlite_nobias.csv.txt')
xornobias <- melt(setDT(xornobias), id.vars = c("X1","epoch"), variable.name = "model")
xornobias$model <- as.character(xornobias$model)
xornobias$model[xornobias$model == 'HSGS'] <- 'BQN'
xornobias$model[xornobias$model == 'phase-encoding\nradius'] <- 'CVQN\nradius'
xornobias$model[xornobias$model == 'phase-encoding\nangle'] <- 'CVQN\nangle'
xornobias$model[xornobias$model == 'phase-encoding\nangleradius'] <- 'CVQN\nangle and radius'
xornobias$model[xornobias$model == 'phase-encoding\noriginal'] <- 'CVQN\noriginal'


p1 = ggplot(xornobias, aes(x=epoch, y=value, group=model, color=model)) +
  geom_line(size=1) +
  scale_color_viridis(discrete = TRUE) +
  labs(x='Epoch', y="Epoch Error", title='Without Bias')+
  theme_minimal()+
  theme(legend.title = element_blank(), 
        legend.text = element_text(size=11))+
  guides(color = guide_legend(override.aes = list(size = 4) ) )

p1

xorbias = read_csv('results/epoch_error_xorlite_bias.csv.txt')
xorbias <- melt(setDT(xorbias), id.vars = c("X1","epoch"), variable.name = "model")
xorbias$model <- as.character(xorbias$model)
xorbias$model[xorbias$model == 'HSGS'] <- 'BQN'
xorbias$model[xorbias$model == 'phase-encoding\nradius'] <- 'CVQN\nradius'
xorbias$model[xorbias$model == 'phase-encoding\nangle'] <- 'CVQN\nangle'
xorbias$model[xorbias$model == 'phase-encoding\nangleradius'] <- 'CVQN\nangle and radius'
xorbias$model[xorbias$model == 'phase-encoding\noriginal'] <- 'CVQN\noriginal'

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


###----- epoch error V2

xornobias = read_csv('data_and_results/XOR/error_by_epoch_nobias.csv')


p1 = ggplot(xornobias, aes(x=epoch, y=value, group=model, color=model)) +
  geom_line(size=1) +
  scale_color_viridis(discrete = TRUE) +
  labs(x='Epoch', y="Epoch Error", title='Without Bias')+
  theme_minimal()+
  scale_y_continuous(limits=c(0,4))+
  theme(legend.title = element_blank(), 
        legend.text = element_text(size=11))+
  guides(color = guide_legend(override.aes = list(size = 4) ) )
p1


xorbias = read_csv('data_and_results/XOR/error_by_epoch_bias.csv')

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


#==============
# diabetes
#=============

d1 = read_csv('results/experiment_diabetes_original.csv')
d2 = read_csv('results/experiment_diabetes_angle.csv')
d3 = read_csv('results/experiment_diabetes_angleradius.csv')
d4 = read_csv('results/experiment_diabetes_radius.csv')
d44 = read_csv('results/experiment_diabetes_radius2.csv')

diabetes = rbind(d1, d2, d3, d4, d44)

diabetes$phase_strategy = paste0("phase-encoding\n", diabetes$phase_strategy)

diabetes$phase_strategy[diabetes$model == 'HSGS'] = 'HSGS'

aggregate(diabetes$avg_error, by=list(diabetes$phase_strategy), mean)

ggplot(diabetes, aes(x=phase_strategy, y=avg_error, fill=phase_strategy)) + 
  geom_boxplot()+
  scale_fill_viridis(name = '', discrete = T)+
  scale_y_continuous(limits = c(0,1))+
  labs(x='Quantum Neuron Model', y="Error")+
  theme_minimal()+ 
  theme(legend.position = "none") 

ggsave('results/diabetes.png', height = 4, width = 7, units = 'in')  


