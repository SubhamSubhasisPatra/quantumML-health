
library(readr)
library(viridis)
library(dplyr)
library(ggplot2)
library(ggpubr)


boxplotResults <- function(dataname, xname, yname, title=''){
  
  df1 <- read_csv(paste0("results/", dataname, '.csv') )
  
  df1$acc = 1 - df1$error
  df1$model[df1$model == 'error_encoding_input'] <- 'RIQN'
  df1$model[df1$model == 'error_phase_encoding'] <- 'RIWQN'
  df1$model[df1$model == 'error_encoding_weight'] <- 'RWQN'
  df1$model[df1$model == 'error_HSGS'] <- 'BWQN'
  #df1 = df1[df1$model != 'error_HSGS',]
  
plot = ggplot(data=df1, aes(x=model, y=acc, fill=model)) +
        geom_boxplot() +
        scale_fill_viridis_d(option = "inferno") +
        scale_y_continuous(limits = c(0, 1))+
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



#==========================
# V5
#==========================


x1 = boxplotResults('version5/experiments_biased_dataframe', 'with bias', 'Accuracy')
x1

x2= boxplotResults('version5/experiments_unbiased_dataframe', 'without bias', 'Accuracy')
x2

d2 = ggarrange(x1, x2, ncol = 2, nrow = 1)
d2 = annotate_figure(d2, top = text_grob('Dataset 2x2', color = "black", face = "bold", size = 14))

ggsave(paste0('results/version5/left_line_experiments.png'), d2, height = 6, width = 8, units = 'in')


#==========================
# V5
#==========================

