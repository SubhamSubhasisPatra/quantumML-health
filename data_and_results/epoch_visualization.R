
library(readr)
library(viridis)
library(dplyr)
library(ggplot2)
library(ggpubr)
library(stringr)
library(viridis)
library(data.table)


xornobias = rbind(read_csv('data_and_results/XOR/error_by_epoch_nobias.csv'), 
                  read_csv('data_and_results/XOR/error_by_epoch_nobias_encoding.csv')[1:60,])


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


xorbias = rbind(read_csv('data_and_results/XOR/error_by_epoch_bias.csv'), 
                  read_csv('data_and_results/XOR/error_by_epoch_bias_encoding.csv')[1:60,])

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
ggsave('data_and_results/XOR/xor-epoch.png', height = 5, width = 7, units = 'in')  
