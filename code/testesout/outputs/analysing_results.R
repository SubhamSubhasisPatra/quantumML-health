
library(readr)
library(viridis)
library(dplyr)
library(ggplot2)
library(ggpubr)


boxplotResults <- function(dataname, xname, yname, title=''){
  
  df1 <- read_csv(paste0("Mestrado/qneuronreal/testesout/outputs/datasets_accuracy/", dataname, '.csv') )
  df1$model[df1$model == 'error_classic'] <- 'RWCN'
  df1$model[df1$model == 'error_classic_bin'] <- 'BWCN'
  df1$model[df1$model == 'error_encoding'] <- 'RWQN'
  df1$model[df1$model == 'error_HSGS'] <- 'BWQN'
  
plot = ggplot(data=df1, aes(x=model, y=error, fill=model)) +
        geom_boxplot() +
        scale_fill_viridis_d(option = "inferno") +
        scale_y_continuous(limits = c(0, 0.5))+
        #geom_jitter(color="black", size=0.4, alpha=0.9) +
        theme_minimal() +
        theme(
          legend.position="none",
          plot.title = element_text(size=11, face='plain'),
          axis.text.x = element_text(size=8,
                                     angle = 13, 
                                     hjust=0.5,
                                     vjust=1,
                                     face="plain")
        ) +
        labs(x=xname, y=yname, title=title)


  return(plot)
}

#=============
# DATASET 2
#=============

x1 = boxplotResults('dataset2_experiments_bias_1noise', 'with bias', 'error')
y1 = boxplotResults('dataset2_experiments_original_1noise', 'without bias', '', '1 noise')

x2 = boxplotResults('dataset2_experiments_bias_2noises', 'with bias', 'error')
y2 = boxplotResults('dataset2_experiments_original_2noises', 'without bias', '', '2 noises')


x3 = boxplotResults('dataset2_experiments_bias_3noises', 'with bias', 'error')
y3 = boxplotResults('dataset2_experiments_original_3noises', 'without bias', '', '3 noises')

d2 = ggarrange(y1, x1, y2, x2, y3, x3, ncol = 2, nrow = 3)
d2 = annotate_figure(d2, top = text_grob('X and Cross (0) vs Square (1)', color = "black", face = "bold", size = 14))
 
ggsave(paste0('Mestrado/qneuronreal/testesout/outputs/plots/dataset2_experiments.png'), d2, height = 7, width = 5, units = 'in')


# DATASET 3
#=====

x1 = boxplotResults('dataset3_experiments_bias_1noise', 'with bias', 'error')
y1 = boxplotResults('dataset3_experiments_original_1noise', 'without bias', '', '1 noise')

x2 = boxplotResults('dataset3_experiments_bias_2noises', 'with bias', 'error')
y2 = boxplotResults('dataset3_experiments_original_2noises', 'without bias', '', '2 noises')


x3 = boxplotResults('dataset3_experiments_bias_3noises', 'with bias', 'error')
y3 = boxplotResults('dataset3_experiments_original_3noises', 'without bias', '', '3 noises')

d3 = ggarrange(y1, x1, y2, x2, y3, x3, ncol = 2, nrow = 3)
d3 = annotate_figure(d3, top = text_grob('X and Square (0) vs Cross', color = "black", face = "bold", size = 14)  )

ggsave(paste0('Mestrado/qneuronreal/testesout/outputs/plots/dataset3_experiments.png'), d3,  height = 7, width = 5, units = 'in')


# DATASET 4
#=====

x1 = boxplotResults('dataset4_experiments_bias_1noise', 'with bias', 'error')
y1 = boxplotResults('dataset4_experiments_original_1noise', 'without bias', '', '1 noise')

x2 = boxplotResults('dataset4_experiments_bias_2noises', 'with bias', 'error')
y2 = boxplotResults('dataset4_experiments_original_2noises', 'without bias', '', '2 noises')


x3 = boxplotResults('dataset4_experiments_bias_3noises', 'with bias', 'error')
y3 = boxplotResults('dataset4_experiments_original_3noises', 'without bias', '', '3 noises')

d4 = ggarrange(y1, x1, y2, x2, y3, x3, ncol = 2, nrow = 3)
d4 = annotate_figure(d4, top = text_grob('X (0) vs Square and Cross (1)', color = "black", face = "bold", size = 14))

ggsave(paste0('Mestrado/qneuronreal/testesout/outputs/plots/dataset4_experiments.png'), d4, height = 7, width = 5, units = 'in')

#============

d_em = ggarrange(d2, d3, d4, ncol = 3, nrow = 1)
d_em
ggsave(paste0('Mestrado/qneuronreal/testesout/outputs/plots/datasets_2_3_4_experiments.png'), d_em, height = 13, width = 13, units = 'in')

# DATASET 5
#=====

x1 = boxplotResults('dataset5_experiments_bias_1noise', 'with bias', 'error')
y1 = boxplotResults('dataset5_experiments_original_1noise', 'without bias', '', '1 noise')

x2 = boxplotResults('dataset5_experiments_bias_2noises', 'with bias', 'error')
y2 = boxplotResults('dataset5_experiments_original_2noises', 'without bias', '', '2 noises')

x3 = boxplotResults('dataset5_experiments_bias_3noises', 'with bias', 'error')
y3 = boxplotResults('dataset5_experiments_original_3noises', 'without bias', '', '3 noises')

ggarrange(y1, x1, y2, x2, y3, x3, ncol = 2, nrow = 3)
ggsave(paste0('Mestrado/qneuronreal/testesout/outputs/plots/dataset5_experiments.png'), height = 6, width = 6, units = 'in')


#===============================================
# ERROR BY EPOCH
#=====================================================


melt_plot <- function(x, title='', legend=FALSE){
  
  # melt data
  data <- melt(setDT(x), id.vars = c("X1"), variable.name = "error")
  colnames(data) = c('epoch', 'Model', 'error')
  
  if (legend == TRUE){
      plot = ggplot(data=data, aes(x = epoch, y=error, group=Model, color=Model))+
      geom_line(size=1)+
      labs(title=title)+
      theme_minimal()+
      theme(plot.title = element_text(size=11, face='plain'))+
      scale_y_continuous(limits = c(0,4))
  } else {
    plot = ggplot(data=data, aes(x = epoch, y=error, group=Model, color=Model))+
      geom_line(size=1)+
      labs(title=title)+
      theme_minimal()+
      theme(legend.position="none",
            plot.title = element_text(size=11, face='plain'))+
      scale_y_continuous(limits = c(0,4))
    
  }
  

  
  return(plot)
}

library(readr)
library(data.table)
library(ggplot2)
library(ggpubr)


#####################3

dataset2_1noise_nobias <- data.frame(read_csv("Mestrado/qneuronreal/testesout/outputs/error_by_epoch_dataset2_1noise_nobias.csv"))[0:20,]
dataset2_1noise_bias <- data.frame(read_csv("Mestrado/qneuronreal/testesout/outputs/error_by_epoch_dataset2_1noise_bias.csv"))[0:20,]

dataset3_1noise_nobias <- data.frame(read_csv("Mestrado/qneuronreal/testesout/outputs/error_by_epoch_dataset3_1noise_nobias.csv"))[0:20,]
dataset3_1noise_bias <- data.frame(read_csv("Mestrado/qneuronreal/testesout/outputs/error_by_epoch_dataset3_1noise_bias.csv"))[0:20,]

dataset4_1noise_nobias <- data.frame(read_csv("Mestrado/qneuronreal/testesout/outputs/error_by_epoch_dataset4_nobias.csv"))[0:20,]
dataset4_1noise_bias <- data.frame(read_csv("Mestrado/qneuronreal/testesout/outputs/error_by_epoch_dataset4_bias.csv"))[0:20,]

dataset5_1noise_nobias <- data.frame(read_csv("Mestrado/qneuronreal/testesout/outputs/error_by_epoch_dataset5_1noise_nobias.csv"))[0:20,]
dataset5_1noise_bias <- data.frame(read_csv("Mestrado/qneuronreal/testesout/outputs/error_by_epoch_dataset5_1noise_bias.csv"))[0:20,]


p1 <- melt_plot(dataset2_1noise_nobias, 'without bias', legend = T)
p2 <- melt_plot(dataset2_1noise_bias, 'with bias', legend = T)
a1 = ggarrange(p1, p2, nrow=1, ncol = 2, common.legend = T)
a1 = annotate_figure(a1, top = text_grob("X and Cross (0) vs Square (1)", color = "black", face = "bold", size = 12))

p3 <- melt_plot(dataset3_1noise_nobias, 'without bias')
p4 <- melt_plot(dataset3_1noise_bias, 'with bias')
a2 = ggarrange(p3, p4, nrow=1, ncol = 2)
a2 = annotate_figure(a2, top = text_grob("X and Square (0) vs Cross (1)", color = "black", face = "bold", size = 12))

p5 <- melt_plot(dataset4_1noise_nobias, 'without bias')
p6 <- melt_plot(dataset4_1noise_bias, 'with bias')
a3 = ggarrange(p5, p6, nrow=1, ncol = 2)
a3 = annotate_figure(a3, top = text_grob("X (0) vs Square and Cross (1)", color = "black", face = "bold", size = 12))

p7 <- melt_plot(dataset5_1noise_nobias, 'without bias')
p8 <- melt_plot(dataset5_1noise_bias, 'with bias')
a4 = ggarrange(p7, p8, nrow=1, ncol = 2)
a4 = annotate_figure(a4, top = text_grob("X (0) vs Square (1) vs Cross (2)", color = "black", face = "bold", size = 12))

ggarrange(a1, a2, a3, a4, nrow=4, ncol = 1, common.legend = T)


ggsave(paste0('Mestrado/qneuronreal/testesout/outputs/plots/EPOCH.png'), height = 10, width = 7, units = 'in')



