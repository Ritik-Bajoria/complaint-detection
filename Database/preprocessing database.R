library(readr)
library(dplyr)

#read the databases and know its columns
dataset1 <- read.csv("C://Users//Legion//Ritik//Desktop//Programming//Intern work//07-Intern//complaint detector//Database//rows.csv")
summary(dataset1)

dataset2 <- read.csv("C://Users//Legion//Ritik//Desktop//Programming//Intern work//04-Intern//Sentiment Analysis//Database//IMDB Dataset.csv")
print(colnames(dataset2))
summary(dataset2)

dataset1 <- head(dataset1, 100000)
dataset2 <- head(dataset2, 40000)

summary(dataset1)
summary(dataset2)
# remove nulls from dataset
na.omit(dataset1)
na.omit(dataset2)

# mapping all positive issues to a with label set as complaint for each row
a <- dataset1 %>%
  mutate(label="complaint") %>%
  mutate(text= paste(Issue,Sub.issue)) %>%
  select(text, label)
print(head(a))


b <- dataset2 %>%
  filter(sentiment %in% c("positive")) %>%
  select(text) %>%
  mutate(label="non-complaint")
print(head(b))

dataset3 <- rbind(a,b)
print(head(dataset3))
# Shuffle the rows
dataset3 <- dataset3[sample(nrow(dataset3)), ]
write.csv(dataset3,"C://Users//Legion//Ritik//Desktop//Programming//Intern work//07-Intern//complaint detector//Database//newdata.csv")
